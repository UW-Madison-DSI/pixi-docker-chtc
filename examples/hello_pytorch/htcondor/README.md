# Running the Hello PyTorch example on CHTC GPUs with Linux containers

Examples are provided for running on GPUs with both Docker containers (under `docker/`) and Apptainer containers (under `apptainer/`).

## Resources

CHTC's documentation has good resources on how to effectively use GPU resources on CHTC.

* [Use GPUS](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs)
* [Explore and Test Docker Containers](https://chtc.cs.wisc.edu/uw-research-computing/docker-test.html)
* [Use an Apptainer Container in HTC Jobs](https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc#use-an-apptainer-container-in-htc-jobs)
* [HTCondor Documentation "Commands for Matchmaking" section with GPU specific arguments](https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html#gpus_minimum_memory)
* [Modal's GPU Glossary section on 'Compute Capability'](https://modal.com/gpu-glossary/device-software/compute-capability#gpu-glossary)

## HTCondor tips

> [!TIP]
> Here are some (perhaps) lesser known HTCondor command line operations that can be useful when planning out HTCondor submit description files and monitoring jobs.

### condor_status

You can use [`condor_status`](https://htcondor.readthedocs.io/en/main/man-pages/condor_status.html) to get information on the kinds of GPUs that exist on the system for use.

```console
$ condor_status -const 'GPUs > 0' -af GPUs_DeviceName GPUs_Capability GPUs_DriverVersion | sort | uniq -c
      6 NVIDIA A100-SXM4-40GB 8.0 12.4
      3 NVIDIA A100-SXM4-40GB 8.0 12.8
     26 NVIDIA A100-SXM4-80GB 8.0 12.4
      8 NVIDIA A100-SXM4-80GB 8.0 12.8
      1 NVIDIA A30 8.0 12.4
      1 NVIDIA A40 8.6 12.4
      3 NVIDIA GeForce GTX 1080 Ti 6.1 12.4
     31 NVIDIA GeForce RTX 2080 Ti 7.5 12.4
      2 NVIDIA GeForce RTX 2080 Ti 7.5 12.8
     13 NVIDIA H100 80GB HBM3 9.0 12.4
     16 NVIDIA H200 9.0 12.8
     11 NVIDIA L40 8.9 12.4
     41 NVIDIA L40 8.9 12.8
     41 NVIDIA L40S 8.9 12.8
      1 Quadro RTX 6000 7.5 12.8
      1 Quadro RTX 8000 7.5 12.4
     10 Tesla P100-PCIE-16GB 6.0 12.4
      1 undefined 7.5 12.8
```

If you wanted to do some `awk` formatting you can get a Markdown table from the output

```
condor_status -const 'GPUs > 0' -af GPUs_DeviceName GPUs_Capability GPUs_DriverVersion | sort | uniq -c | awk 'BEGIN {print "| Count | GPU Model | Capability | Driver Version |"; print "|-------|-----------|------------|----------------|"} {count=$1; driver=$NF; capability=$(NF-1); $1=""; $NF=""; $(NF-1)=""; gsub(/^[ \t]+|[ \t]+$/, ""); printf "| %5s | %-30s | %s | %s |\n", count, $0, capability, driver}'
```

| Count | GPU Model | Capability | Driver Version |
|-------|-----------|------------|----------------|
|     6 | NVIDIA A100-SXM4-40GB          | 8.0 | 12.4 |
|     3 | NVIDIA A100-SXM4-40GB          | 8.0 | 12.8 |
|    26 | NVIDIA A100-SXM4-80GB          | 8.0 | 12.4 |
|     8 | NVIDIA A100-SXM4-80GB          | 8.0 | 12.8 |
|     1 | NVIDIA A30                     | 8.0 | 12.4 |
|     1 | NVIDIA A40                     | 8.6 | 12.4 |
|     3 | NVIDIA GeForce GTX 1080 Ti     | 6.1 | 12.4 |
|    31 | NVIDIA GeForce RTX 2080 Ti     | 7.5 | 12.4 |
|     2 | NVIDIA GeForce RTX 2080 Ti     | 7.5 | 12.8 |
|    13 | NVIDIA H100 80GB HBM3          | 9.0 | 12.4 |
|    16 | NVIDIA H200                    | 9.0 | 12.8 |
|    11 | NVIDIA L40                     | 8.9 | 12.4 |
|    41 | NVIDIA L40                     | 8.9 | 12.8 |
|    42 | NVIDIA L40S                    | 8.9 | 12.8 |
|     1 | Quadro RTX 6000                | 7.5 | 12.8 |
|     1 | Quadro RTX 8000                | 7.5 | 12.4 |
|    10 | Tesla P100-PCIE-16GB           | 6.0 | 12.4 |
|     1 | undefined                      | 7.5 | 12.8 |

If you want to avoid [pre-Turing architecture GPUs for stability](https://github.com/conda-forge/cudnn-feedstock/issues/124), you can also filter on NVIDIA GPU Compute Capability

```console
$ condor_status -const 'GPUs > 0' -const 'GPUs_Capability >= 7.0' -af GPUs_DeviceName GPUs_Capability GPUs_DriverVersion | sort | uniq -c
      6 NVIDIA A100-SXM4-40GB 8.0 12.4
      3 NVIDIA A100-SXM4-40GB 8.0 12.8
     26 NVIDIA A100-SXM4-80GB 8.0 12.4
      8 NVIDIA A100-SXM4-80GB 8.0 12.8
      1 NVIDIA A30 8.0 12.4
      1 NVIDIA A40 8.6 12.4
     31 NVIDIA GeForce RTX 2080 Ti 7.5 12.4
      2 NVIDIA GeForce RTX 2080 Ti 7.5 12.8
     13 NVIDIA H100 80GB HBM3 9.0 12.4
     15 NVIDIA H200 9.0 12.8
     11 NVIDIA L40 8.9 12.4
     41 NVIDIA L40 8.9 12.8
     42 NVIDIA L40S 8.9 12.8
      1 Quadro RTX 6000 7.5 12.8
      1 Quadro RTX 8000 7.5 12.4
      1 undefined 7.5 12.8
```

### condor_watch_q

[`condor_watch_q`](https://htcondor.readthedocs.io/en/main/man-pages/condor_watch_q.html) will continually monitor and track the state of jobs.
This can be useful if you want to monitor the state of multiple jobs that are in various stages, or if you want to monitor the status of a single job more closely without running `condor_q` repeatedly.

```console
$ condor_watch_q  # this will update the terminal each second
BATCH         IDLE  RUN  DONE  TOTAL  JOB_IDS
ID: 2560590     -    1     -      1   2560590.0 [=============================================================================]

[=============================================================================]

Total: 1 jobs; 1 running

Updated at 2025-11-06 23:54:09
Press any key to exit
```

which eventually resolves to job completion

```console
$ condor_watch_q
BATCH         IDLE  RUN  DONE  TOTAL  JOB_IDS
ID: 2560590     -    -     1      1          [#############################################################################]

[#############################################################################]

Total: 1 jobs; 1 completed

Updated at 2025-11-07 00:02:13
Press any key to exit
```
