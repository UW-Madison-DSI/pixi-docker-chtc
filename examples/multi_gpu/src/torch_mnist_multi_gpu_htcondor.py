import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    # HTCondor sets CUDA_VISIBLE_DEVICES, but we need to set up process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device based on rank
    torch.cuda.set_device(rank)

    return rank


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_epoch(model, device, train_loader, optimizer, epoch, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if rank == 0 and batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    if rank == 0:
        print(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test(model, device, test_loader, rank):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    if rank == 0:
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )


def main(rank, world_size):
    """Main training function for each process"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 18
    lr = 0.01

    # Data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # IMPORTANT: Only rank 0 downloads the dataset to avoid race conditions
    if rank == 0:
        print("Rank 0: Downloading MNIST dataset...")
        datasets.MNIST("./data", train=True, download=True, transform=transform)
        datasets.MNIST("./data", train=False, download=True, transform=transform)
        print("Rank 0: Dataset download complete")

    # Synchronize all processes - wait for rank 0 to finish downloading
    dist.barrier()

    # Now all ranks can safely load the dataset
    train_dataset = datasets.MNIST(
        "./data", train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=False, transform=transform
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Only rank 0 needs the test loader
    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # Create model and move to GPU
    model = CNN().to(device)

    # Wrap model with DDP
    # IMPORTANT: Use device 0 since HTCondor may set CUDA_VISIBLE_DEVICES per process
    # Check if we should use rank or 0 for device_ids
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," in visible_devices:
        # Multiple GPUs visible, use rank
        model = DDP(model, device_ids=[rank])
    else:
        # Single GPU visible per process (HTCondor style), use device 0
        model = DDP(model, device_ids=[0])

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    # Training loop
    if rank == 0:
        print(f"Training on {world_size} GPUs")
        print(f"Total training samples: {len(train_dataset)}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Effective batch size: {batch_size * world_size}\n")

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        train_epoch(model, device, train_loader, optimizer, epoch, rank)

        if rank == 0 and test_loader is not None:
            test(model, device, test_loader, rank)

    # Save model (only rank 0)
    if rank == 0:
        file_name = "mnist_cnn.pt"
        torch.save(model.module.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    import sys

    # Get rank and world_size from command line arguments
    # These will be set by the wrapper script
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])

    main(rank, world_size)
