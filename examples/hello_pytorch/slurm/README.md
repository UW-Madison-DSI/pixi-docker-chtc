# Running the Hello PyTorch example on GPUs on SLURM clusters

This example contains the files:

* `hello_pytorch_gpu_slurm.sh`: The SLURM job submission file
* `execute.sh`: The execute shells script that contains the commands that are run as the job
* `submit.sh`: The shell script that submits `hello_pytorch_gpu_slurm.sh` to SLURM as a standard batch job with `sbatch`

## Running

After logging onto the SLURM cluster and cloning this repository, navigate to this directory and run

```
bash submit.sh
```

This will submit the job.
After the job finishes, it will return the job's `stdout` and `stderr` files which contain the job allocation number, or `JOBID`:


* `slurm-$(JOBID).out`
* `slurm-$(JOBID).err`

## Checking status

```
squeue --user $USER
```
