# Running the Hello PyTorch example on CHTC GPUs with Apptainer `pixi-pack`

This example contains the files:

* `hello_pytorch_gpu_apptainer.sub`: The HTCondor job submission file
* `hello_pytorch_gpu_apptainer.sh`: The execute shells script that contains the commands that are run as the job
* `submit.sh`: The shell script that submits `hello_pytorch_gpu_apptainer.sub` to HTCondor as a standard batch job with `condor_submit`
* `interact.sh`: The shell script that submits `hello_pytorch_gpu_apptainer.sub` to HTCondor as an interactive job with `condor_submit -interactive`

## Using Apptainer as an alternative to Docker

CHTC is rare as it provides users the ability to run Docker directly.
Most HTC and HPC systems do not allow users to use Docker given security risks and instead use Apptainer.
In most situations, Apptainer is able to automatically convert a Docker image, or other Open Container Initiative (OCI) container image format, to Apptainer's [Singularity Image Format](https://github.com/apptainer/sif) `.sif` container image format, and so no additional work is required.
However, the overlay system of Apptainer is different from Docker, which means that the `ENTRYPOINT` of a Docker container image might not get correctly translated into an Apptainer `runscript` and `startscript`.
In might be advantageous, depending on your situation, to instead write an [Apptainer `.def` definition file](https://apptainer.org/docs/user/main/definition_files.html), giving full control over the commands, and then build that `.def` file into an `.sif` Apptainer container image.

```
apptainer build pixi-docker-chtc.sif oras://ghcr.io/uw-madison-dsi/pixi-docker-chtc:apptainer-hello-pytorch-noble-cuda-12.9
apptainer run --nv --containall pixi-docker-chtc.sif bash
```

To mimic the conditions of CHTC jobs that use Apptainer you can test with

```
apptainer exec \
    --scratch /tmp \
    --scratch /var/tmp \
    --workdir $(pwd) \
    --pwd $(pwd) \
    --bind $(pwd) \
    --no-home \
    --containall \
    --nv \
    <container image file>.sif bash
```

## Differences from using Docker containers

Unlike Docker jobs, there is no "container universe" for CHTC jobs that use Apptainer, which means that we need to make the `.sif` container image file available to the worker nodes to use.
We do this by placing the container images on CHTC's `/staging/<username>` storage where they are cached with [Pelican](https://pelicanplatform.org/), and transferred to HTCondor jobs using the

```
container_image = osdf:///chtc/staging/<username>/<container image name>.sif
```

HTCondor submission syntax.

**Example**:

```
container_image = osdf:///chtc/staging/mfeickert/apptainer/pixi-docker-chtc-54990e82.sif
```

Pelican offers the advantage of [not limiting your jobs to running on locations where `/staging/` is mounted](https://chtc.cs.wisc.edu/uw-research-computing/scaling-htc#3-submitting-jobs-to-run-beyond-chtc).

> [!WARNING]
> Pelican treats all cached files as immutable, so each file placed on `/staging/` should have a unique name, such as including information about the file's hash in the file name
> ```
> mv pixi-docker-chtc.sif pixi-docker-chtc-"$(openssl sha256 -r pixi-docker-chtc.sif | cut -b -8)".sif
> ```

### Advantages

* Apptainer is the HTC and HPC industry standard for secure Linux containers.
* Apptainer should be supported at all HTC and HPC environments, making workflows easier to port.

### Disadvantages

* As the container image needs to be copied to the worker nodes (there is no additional technology to handle the mounting), the HTCondor job requires substantially larger memory and disk resources than a typical job to be able to unpack the archive into the environment directory tree.
   - The Linux container jobs use

```
request_memory = 2GB
request_disk = 2GB
```

while the Apptainer jobs use

```
request_memory = 4GB
request_disk = 10GB
```

* As the container images needs to get run on the worker node for each job, this can take a substantial amount of time (multiple minutes) before any code execution can begin, and so workflows on Apptainer containers might start slower than Docker containers..

## Resources

* [Manage Large Data in HTC Jobs](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata.html)
* [Use an Apptainer Container in HTC Jobs](https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc#use-an-apptainer-container-in-htc-jobs)
