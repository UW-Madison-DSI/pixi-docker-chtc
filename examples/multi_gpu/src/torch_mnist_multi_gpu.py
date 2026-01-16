import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST Distributed Training Example"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training per GPU (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=125,
        metavar="S",
        help="random seed (default: 125)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="save the trained model (default: True)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        metavar="N",
        help="save checkpoint every N epochs (default: 0, disabled)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="path to checkpoint to resume from",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic mixed precision training (default: False)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="directory for dataset storage (default: ./data)",
    )
    return parser.parse_args()


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


def setup_distributed():
    """Initialize distributed training environment

    Supports both torchrun (recommended) and direct SLURM execution.
    torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE
    SLURM sets: SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS

    Environment variables:
        MASTER_ADDR: Hostname of rank 0 node (set by SLURM script)
        MASTER_PORT: Port for distributed communication (set by SLURM script)

    Note: While torchrun can set MASTER_ADDR/MASTER_PORT internally when using
    --rdzv-endpoint, we require them to be set explicitly in the SLURM script.
    This ensures consistent behavior and allows the script to work both with
    torchrun and with direct SLURM execution (e.g., srun python script.py).
    """
    # Try torchrun environment variables first (standard approach)
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # Verify required environment variables for distributed training.
    # These must be set by the SLURM submission script before launching torchrun.
    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError(
            "MASTER_ADDR must be set for distributed training. "
            "This should be set by the SLURM script before launching torchrun."
        )
    if "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            "MASTER_PORT must be set for distributed training. "
            "This should be set by the SLURM script before launching torchrun."
        )

    # Initialize process group using NCCL backend (optimized for NVIDIA GPUs).
    # init_method="env://" reads MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE
    # from environment variables.
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set the CUDA device for this process based on local rank.
    # local_rank maps to the GPU index on this node (0, 1, 2, ...).
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_epoch(model, device, train_loader, optimizer, epoch, rank, scaler=None):
    """Train for one epoch.

    Args:
        model: The DDP-wrapped model
        device: CUDA device for this process
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        epoch: Current epoch number
        rank: Process rank (0 for main process)
        scaler: GradScaler for mixed precision training (None to disable AMP)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixed precision training with autocast
        # Reference: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with autocast(device_type="cuda", enabled=use_amp):
            output = model(data)
            loss = F.nll_loss(output, target)

        if use_amp:
            # Scale loss and call backward to create scaled gradients
            scaler.scale(loss).backward()
            # Unscale gradients and call optimizer.step()
            scaler.step(optimizer)
            # Update the scale for next iteration
            scaler.update()
        else:
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


def save_checkpoint(model, optimizer, epoch, scaler, filepath):
    """Save training checkpoint.

    Args:
        model: The DDP-wrapped model
        optimizer: Optimizer instance
        epoch: Current epoch number
        scaler: GradScaler instance (may be None)
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, scaler=None):
    """Load training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: The DDP-wrapped model
        optimizer: Optimizer instance
        scaler: GradScaler instance (may be None)

    Returns:
        int: The epoch to resume from
    """
    checkpoint = torch.load(filepath, weights_only=True)
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"]


def main():
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Print distributed configuration (rank 0 only)
    if rank == 0:
        print("\n" + "=" * 60)
        print("Distributed Training Configuration")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Number of nodes: {os.environ.get('SLURM_NNODES', 'N/A')}")
        print(f"GPUs per node: {os.environ.get('SLURM_GPUS_PER_NODE', 'N/A')}")
        print(f"Master address: {os.environ['MASTER_ADDR']}")
        print(f"Master port: {os.environ['MASTER_PORT']}")
        print(f"Backend: nccl")
        print(f"CUDA device: {device}")
        print(f"Mixed precision (AMP): {args.amp}")
        print("=" * 60 + "\n")

    # Data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Download dataset on rank 0 only to avoid race conditions.
    # All other ranks wait at the barrier until download completes.
    # Reference: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    if rank == 0:
        print(f"Rank 0: Downloading MNIST dataset to {args.data_dir}...")
        datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        print("Rank 0: Dataset download complete")

    # Synchronize all processes - wait for rank 0 to finish downloading
    dist.barrier()

    # Now all ranks can safely load the dataset (download=False)
    train_dataset = datasets.MNIST(
        args.data_dir, train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        args.data_dir, train=False, download=False, transform=transform
    )

    # Create distributed sampler for sharding data across processes
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # Data loaders
    # num_workers is automatically read from SLURM_CPUS_PER_TASK to match
    # the allocated CPU resources per task.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    if rank == 0:
        print(f"DataLoader num_workers: {num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Only rank 0 needs the test loader for evaluation
    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Create model and move to GPU
    model = CNN().to(device)

    # Wrap model with DDP for distributed training
    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    # Note on learning rate scaling: With DDP, the effective batch size is
    # batch_size * world_size. The linear scaling rule (Goyal et al., 2017,
    # https://arxiv.org/abs/1706.02677) suggests scaling lr proportionally.
    # However, Adadelta uses per-parameter adaptive learning rates, making it
    # less sensitive to batch size changes. We keep lr fixed here, but for
    # SGD-based optimizers, consider: lr = args.lr * world_size
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Initialize GradScaler for mixed precision training (AMP)
    # Reference: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    scaler = GradScaler() if args.amp else None

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler) + 1
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")

    # Training loop
    if rank == 0:
        print(f"Training on {world_size} GPUs")
        print(f"Total training samples: {len(train_dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for distributed sampler to ensure different shuffling each epoch
        train_sampler.set_epoch(epoch)
        train_epoch(model, device, train_loader, optimizer, epoch, rank, scaler)

        # Synchronize all processes before evaluation.
        # This ensures all ranks have completed training for this epoch.
        dist.barrier()

        if rank == 0 and test_loader is not None:
            test(model, device, test_loader, rank)

        # Save periodic checkpoint (rank 0 only)
        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            if rank == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(model, optimizer, epoch, scaler, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            # Ensure checkpoint is written before other ranks proceed
            dist.barrier()

    # Save final model (only rank 0)
    if args.save_model and rank == 0:
        file_name = "mnist_cnn.pt"
        torch.save(model.module.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    # Cleanup distributed process group
    cleanup_distributed()


if __name__ == "__main__":
    main()
