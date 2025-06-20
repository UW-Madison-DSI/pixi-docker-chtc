# Running the Hello PyTorch example on CHTC GPUs with Apptainer

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

where the [`--nv` flag](https://apptainer.org/docs/user/latest/gpu.html#nvidia-gpus-cuda-standard) supports NVIDIA CUDA devices like GPUs.

## Differences from using Docker containers

Docker is the only supported containerization system under the [container universe](https://htcondor.readthedocs.io/en/latest/users-manual/env-of-job.html#container-universe-jobs) that includes an image cache, and so Docker jobs have the ability to set the [`docker_pull_policy`](https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html#docker_pull_policy).
The `docker_pull_policy` is used on HTCondor [Execution Points](https://htcondor.readthedocs.io/en/latest/admin-manual/ep-policy-configuration.html#configuration-for-execution-points) (EP) that use Docker and is ignored on EPs that use Apptainer.
As HTCondor does not provide its own cache management for container images it can transfer a `.sif` file or expanded SIF directory from an HTCondor [Access Point](https://htcondor.readthedocs.io/en/latest/admin-manual/ap-policy-configuration.html#configuration-for-access-points) (AP) upon request.
(Source: [Email exchange with Jaime Frey in June 2025](https://github.com/UW-Madison-DSI/pixi-docker-chtc/pull/17#discussion_r2153136096).)

### Transferring `.sif` files to HTCondor Execution Points

The transferring of `.sif` files can happen one of two ways.
Both involve setting the `container` universe in the submission file along with giving an endpoint for the `container_image`.

#### Using a remote container registry

If the `.sif` file exists on a container registry then the container `.sif` file can be provided to the `container_image` argument [using the `oras://` prefix](https://htcondor.readthedocs.io/en/latest/users-manual/env-of-job.html#container-universe-jobs).

```
universe = container
container_image = oras://<container registry domain>/<account>/<container name>:<tag>
```

**Example**:

```
universe = container
container_image = oras://ghcr.io/uw-madison-dsi/pixi-docker-chtc:apptainer-hello-pytorch-noble-cuda-12.9
```

#### Transferring from CHTC `/staging`

Container `.sif` files on CHTC's `/staging/<username>` storage are cached with [ODSF](https://osg-htc.org/services/osdf), and transferred to HTCondor jobs using the

```
container_image = osdf:///chtc/staging/<username>/<container image name>.sif
```

HTCondor submission syntax.

**Example**:

```
container_image = osdf:///chtc/staging/mfeickert/apptainer/pixi-docker-chtc-54990e82.sif
```

OSDF offers the advantage of [not limiting your jobs to running on locations where `/staging/` is mounted](https://chtc.cs.wisc.edu/uw-research-computing/scaling-htc#3-submitting-jobs-to-run-beyond-chtc).

> [!WARNING]
> OSDF and Pelican treat all cached files as immutable, so each file placed on `/staging/` should have a unique name, such as including information about the file's hash in the file name
> ```
> mv pixi-docker-chtc.sif pixi-docker-chtc-"$(openssl sha256 -r pixi-docker-chtc.sif | cut -b -8)".sif
> ```

### Advantages

* Apptainer is the HTC and HPC industry standard for secure Linux containers.
* Apptainer should be supported at all HTC and HPC environments, making workflows easier to port across facilities.

### Disadvantages

* The HTCondor `container` universe does not have a supported image cache for Apptainer `.sif` files, and HTCondor does not provide its own cache management for container images, and so if a remote container registry is used the container files will not be cached and will be repulled.
The can make some jobs slower to start to run than Docker universe jobs.
* If the OSDF `/staging/` cache is used, the user is now responsible for managing the transferring of `.sif` container files to the `/staging/` storage and providing unique file names for them to avoid issues with OSDF's immutable cache.

## Resources

* [Manage Large Data in HTC Jobs](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata.html)
* [Use an Apptainer Container in HTC Jobs](https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc#use-an-apptainer-container-in-htc-jobs)
