# Running the Hello PyTorch example on GPUs on SLURM clusters

This example contains the files:

* `hello_pytorch_gpu_slurm.sh`: The SLURM job submission file
* `execute.sh`: The execute shells script that contains the commands that are run as the job
* `submit.sh`: The shell script that submits `hello_pytorch_gpu_slurm.sh` to SLURM as a standard batch job with `sbatch`

## Configuration

Note that in the `hello_pytorch_gpu_slurm.sh` there will be cluster specific information that needs to be adjusted such as:

* `partition`
* `account`
* `constraint`

For example, if you have [NSF ACCESS](https://access-ci.org/) credits and are running on [NCSA's DELTA](https://docs.ncsa.illinois.edu/systems/delta/en/latest/) these parameters [might look like](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html#gpu-on-a-compute-node):

* `partition=gpuA40x4`
* `account` (match to an "Account" returned by the `accounts` command)
* `constraint="scratch"`

## Running

After logging onto the SLURM cluster and cloning this repository, navigate to this directory and run

```
bash submit.sh
```

This will submit the job.
After the job finishes, it will return the job's `stdout` and `stderr` files which contain the job allocation number, or `JOBID`:


* `hello_pytorch.slurm-$(JOBID).out`
* `hello_pytorch.slurm-$(JOBID).err`

## Checking status

```
squeue --user $USER
```
