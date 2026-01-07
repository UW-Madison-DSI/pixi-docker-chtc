#!/usr/bin/env python3
"""
Simple MPI ring communication example.
Each process sends a message to the next process in a ring topology.
"""

from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each process sends to the next, last sends to first (ring)
    next_rank = (rank + 1) % size
    prev_rank = (rank - 1 + size) % size

    # Create a message that includes this process's rank
    send_data = f"Hello from rank {rank}"

    print(f"Rank {rank}/{size}: Sending '{send_data}' to rank {next_rank}", flush=True)

    # Send to next, receive from previous
    recv_data = comm.sendrecv(sendobj=send_data, dest=next_rank,
                              source=prev_rank)

    print(f"Rank {rank}/{size}: Received '{recv_data}' from rank {prev_rank}", flush=True)

    # Barrier to ensure all processes complete
    comm.Barrier()

    if rank == 0:
        print(f"\nRing communication completed successfully across {size} nodes!", flush=True)

if __name__ == "__main__":
    main()
