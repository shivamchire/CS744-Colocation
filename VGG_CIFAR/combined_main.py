import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from performance_iterator import PerformanceIterator

def train_vgg19_cifar10(args, num_steps=50, lr=0.001, momentum=0.9, batch_size=20):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    if args.enable_perf_log:
        trainloader = PerformanceIterator(trainloader, None, None, None, args.log_file)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load pre-trained VGG-19 model
    model = models.vgg19(pretrained=True)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Train the model
    running_loss = 0.0
    with tqdm(total=num_steps) as pbar:
        train_iter = iter(trainloader)
        for iteration in range(1, num_steps + 1):
            # Get the inputs
            try:
                inputs, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(trainloader)
                inputs, labels = next(train_iter)

            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if iteration % 50 == 0:
                pbar.set_postfix({'loss': running_loss / 50})
                pbar.update(50)
                running_loss = 0.0
    torch.save(model.state_dict(), 'vgg19_cifar10.pth')
    print('Finished Training')

def inference_vgg19_cifar10(args, model_path='./vgg19_cifar10.pth', num_test_instances=None, batch_size=20):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    # Use specified number of test instances if provided, otherwise use all instances
    if num_test_instances:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2,
                                                 sampler=torch.utils.data.RandomSampler(testset, num_samples=num_test_instances))
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
    if args.enable_perf_log:
        testloader = PerformanceIterator(testloader, None, None, None, args.log_file)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load the saved VGG-19 model
    model = models.vgg19(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # Perform inference on the test set
    correct = 0
    total = 0
    total_steps = 0
    while total_steps < args.num_steps:
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_steps += 1
                if total_steps > args.num_steps:
                    break

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f} %')

def main(args):
    job_type = args.job_type
    num_steps = args.num_steps
    lr = args.lr
    momentum = args.momentum
    if job_type == 'training':
        train_vgg19_cifar10(args, num_steps=num_steps, lr=lr, momentum=momentum, batch_size=args.batch_size)
    elif job_type == 'inference':
        inference_vgg19_cifar10(args, model_path=args.model_path, num_test_instances=num_steps, batch_size=args.batch_size)
    else:
        print("Invalid job_type. Please specify 'training' or 'inference'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG-19 on CIFAR-10 dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (default: 0.9)")
    parser.add_argument("--model_path", type=str, default='./vgg19_cifar10.pth', help="Path to the saved model file (default: './model.pt')")
    parser.add_argument("--num_test_instances", type=int, default=None, help="Number of test instances to use for inference (default: use all instances)")
    parser.add_argument("--job_type", type=str, default='training', choices=['training', 'inference'], help="Specify 'training' or 'inference'")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size. (default:20)")
    parser.add_argument("--enable_perf_log", action='store_true', default=False, help="If set, enable performance logging")
    parser.add_argument("--log_file", type=str, default="vgg.log", help="Log file name(default:vgg.log)")

    args = parser.parse_args()
    main(args)

