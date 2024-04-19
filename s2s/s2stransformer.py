from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from torch.utils.data import DataLoader
from performance_iterator import PerformanceIterator

from typing import Iterable, List
import argparse, random, numpy as np, torch, time, subprocess

from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence

import warnings

# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
DEVICE = 'cpu'

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Placeholders
token_transform = {}
vocab_transform = {}
text_transform = {}

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def processDataset():
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor

# Adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# Convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Multi30k test set is unavailable so we evaluate on validation set
def evaluate(model, batchSize, numBatches, lossFn=None, logFile=None):
    model.eval()
    # losses = 0
    correct_predictions = 0
    total_predictions = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=batchSize, collate_fn=collate_fn)
    if logFile is not None:
        val_dataloader = PerformanceIterator(val_dataloader, None, None, None, logFile)
    valIterator = iter(val_dataloader)

    testStart = time.time()
    for i in range(1, numBatches+1):
        try:
            src, tgt = next(valIterator)
        except StopIteration:
            valIterator = iter(val_dataloader)
            src, tgt = next(valIterator)

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        predicted_tokens = logits.argmax(dim=-1)
        correct_predictions += (predicted_tokens == tgt[1:, :]).sum().item()
        total_predictions += (tgt[1:, :] != PAD_IDX).sum().item()

        # tgt_out = tgt[1:, :]
        # loss = lossFn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # losses += loss.item()
    testTime = time.time() - testStart

    accuracy = correct_predictions / total_predictions
    # avgLoss = losses / len(list(val_dataloader))
    avgTime = testTime / numBatches

    return accuracy, avgTime

def train(model, numBatches, optimizer, lossFn, batchSize, logFile=None):
    model.train()
    losses = 0
    n = 50

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=batchSize, collate_fn=collate_fn)
    if logFile is not None:
        train_dataloader = PerformanceIterator(train_dataloader, None, None, None, logFile)
    trainIterator = iter(train_dataloader)

    totalTrainingTime = time.time()
    for i in range(1, numBatches+1):
        try:
            src, tgt = next(trainIterator)
        except StopIteration:
            trainIterator = iter(train_dataloader)
            src, tgt = next(trainIterator)
        
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = lossFn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        if i % n == 0:
            avgLoss = losses / n
            print(f"Step: {i}, Avg Loss: {avgLoss:.4f}")
            losses = 0.0

    totalTrainingTime = time.time() - totalTrainingTime        
    print(f"Training complete. Total training time: {totalTrainingTime:.4f}s")
    print(f"Average training time per step: {totalTrainingTime/numBatches:.4f}s")

# generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    start = time.time()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    totalTime = time.time() - start
    op = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    return op, totalTime


"""
pip install following
- torchtext
- spacy
- torchdata
- portalocker

run the following
 python3 -m spacy download de_core_news_sm
 python3 -m spacy download en_core_web_sm
"""
if __name__ == '__main__':

    # Download spaCy models
    subprocess.run(['python3', '-m', 'spacy', 'download', 'de_core_news_sm'], stdout=subprocess.DEVNULL)
    subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'], stdout=subprocess.DEVNULL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuIdx', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--batch_size', dest='batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--saveModel', type=str, default="s2s.pt", help='Path to save model')
    parser.add_argument('--job_type', dest='mode', default="training", type=str, help='training/inference')
    parser.add_argument("--enable_perf_log", action='store_true', default=False, help="If set, enable performance logging")
    parser.add_argument("--num_steps", dest='numBatches', type=int, default=1000, help="Number of training steps")
    parser.add_argument("--log_file", type=str, default="transformer.log", help="Log file name(default:vgg.log)")
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpuIdx}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    if args.mode =='training':
        mode = 'train'
    elif args.mode == 'inference':
        mode = 'infer'
    elif args.mode == 'translation':
        mode = 'translate'
    else:
        print("Invalid job type. Use 'training'/'inference'/'translation'")
        exit()

    saveModel = None if args.saveModel is None else args.saveModel
    batchSize = 32 if args.batchSize is None else args.batchSize
    numBatches = 50 if args.numBatches is None else args.numBatches

    set_seed(0)
    processDataset()

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    if mode == 'train':

        alpha = 0.001 if args.alpha is None else args.alpha
        numEpochs = 10 if args.epochs is None else args.epochs
        logFile = args.log_file if args.enable_perf_log else None

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=alpha, betas=(0.9, 0.98), eps=1e-9)
        lossFn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        print(f"Training a Seq2Seq transformer on Multi30k dataset for {numBatches} steps with batch size {batchSize} and learning rate {alpha}")
        train(model, numBatches, optimizer, lossFn, batchSize, logFile)

        if saveModel:
            torch.save(model.state_dict(), saveModel)
            print(f"Model params saved to {saveModel}")
            
        print(translate(model, "Sie spielen Fußball."))

    elif mode == 'translate':
        if saveModel is None:
            print("Model path not provided")
            exit()
        
        model.load_state_dict(torch.load(saveModel))
        model = model.to(DEVICE)
        model.eval()

        while True:
            src_sentence = input("Enter a sentence in German: ")
            if src_sentence == 'exit':
                break
            translated_sentence, inferTime = translate(model, src_sentence)
            print(f"Translated sentence: {translated_sentence}")
            print(f"Inference time: {inferTime:.4f} seconds")

    elif mode == 'infer':
        if saveModel is None:
            print("Model path not provided")
            exit()
        
        logFile = 'inferS2S.log' if args.enable_perf_log else None

        model.load_state_dict(torch.load(saveModel))
        model = model.to(DEVICE)
        model.eval()

        acc, avgTime = evaluate(model, batchSize, numBatches, logFile=logFile)
        print(f"Accuracy: {acc:.4f}")
        print(f"Average inference time per batch: {avgTime:.4f} seconds")

