# coding: utf-8
import argparse
import time
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.onnx
from tqdm import tqdm
import data
import model as mdl
from performance_iterator import PerformanceIterator

eval_batch_size = 10
criterion = nn.NLLLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def evaluate(args, data_source, model, corpus):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(args, data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def get_next_batch(args, source, i, batch_size):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def train_model(args, model, corpus, device, num_steps):
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    criterion = nn.NLLLoss()
    lr = args.lr
    best_val_loss = None

    try:
        i = 0
        with tqdm(total=num_steps) as pbar:
            while i < num_steps:
                iteration_start_time = time.time()
                model.train()
                total_loss = 0.
                start_time = time.time()
                ntokens = len(corpus.dictionary)
                if args.model != 'Transformer':
                    hidden = model.init_hidden(args.batch_size)

                while train_data.size(0) < num_steps * args.bptt:
                    print("Appended train_data to itself")
                    train_data = torch.cat([train_data, train_data], dim=0)

                data_loader = tqdm(range(0, train_data.size(0) - 1, args.bptt))  

                if args.enable_perf_log:
                    data_loader = PerformanceIterator(data_loader, None, None, None, args.log_file)

                for batch_start in data_loader:
                    data, targets = get_next_batch(args, train_data, batch_start, args.batch_size)
                    model.zero_grad()
                    if args.model == 'Transformer':
                        output = model(data)
                        output = output.view(-1, ntokens)
                    else:
                        hidden = repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                    loss = criterion(output, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    for p in model.parameters():
                        p.data.add_(p.grad, alpha=-lr)
                    total_loss += loss.item()
                    if i % args.log_interval == 0 and i > 0:
                        cur_loss = total_loss / args.log_interval
                        elapsed = time.time() - start_time
                        ips = args.log_interval / elapsed #(i * args.batch_size) / elapsed
                        tqdm.write('| iteration {:5d} | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f} | IPS {:6.2f}'.format(
                            i, lr, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), ips))
                        total_loss = 0
                        start_time = time.time()
                    if args.dry_run:
                        break
                    i += 1
                    pbar.update(1)
                    if i >= num_steps:
                        break
                val_loss = evaluate(args, val_data, model, corpus)
                print('-' * 89)
                print('| end of iteration {:5d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(i, (time.time() - iteration_start_time),
                                                val_loss, math.exp(val_loss)))
                print('-' * 89)
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                else:
                    lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def export_onnx(path, device, model, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

# Inference script
def generate_working_without_dataloader(args, device):
    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3.")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)

    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    with open(args.outf, 'w') as outf:
        with torch.no_grad():  # no tracking history
            for i in range(args.num_steps):
                if is_transformer_model:
                    output = model(input, False)
                    word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    input = torch.cat([input, word_tensor], 0)
                else:
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.num_steps))

def generate(args, device):
    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3.")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)

    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'


    # Define the data loader
    dataloader = torch.utils.data.DataLoader(
        corpus.test, batch_size=1, shuffle=False, drop_last=False
    )
    if args.enable_perf_log:
        dataloader = PerformanceIterator(dataloader, None, None, None, args.log_file)

    dataloader = iter(dataloader)

    with open(args.outf, 'w') as outf:
        with torch.no_grad():  # no tracking history
            for i, datas in enumerate(dataloader):
                # Get the input from the data batch
                if i >= args.num_steps:
                    break
                input = datas.to(device)

                if is_transformer_model:
                    output = model(input, False)
                    word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    input = torch.cat([input, word_tensor], 0)
                else:
                    output, hidden = model(input, None)  # No need for hidden with data loader
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i * len(datas), args.num_steps))

        # Handle performance logging if enabled (original code snippet remains)
        

def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
    parser.add_argument('--job_type', type=str, choices=['training', 'inference'], default='training',
                        help='Type of job: training or inference')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--mps', action='store_true', default=False,
                            help='enables macOS GPU training')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dry-run', action='store_true',
                        help='verify the code and the model')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
    parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
    parser.add_argument("--enable_perf_log", action='store_true', default=True, help="If set, enable performance logging")
    parser.add_argument("--log_file", type=str, default="lstm.log", help="Log file name(default:lstm.log)")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if not args.mps:
            print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

    use_mps = args.mps and torch.backends.mps.is_available()
    if args.cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.job_type == 'training':
        corpus = data.Corpus(args.data)
        ntokens = len(corpus.dictionary)
        if args.model == 'Transformer':
            model = mdl.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
        else:
            model = mdl.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

        train_model(args, model, corpus, device, args.num_steps)

    elif args.job_type == 'inference':
        generate(args, device)

if __name__ == "__main__":
    main()
