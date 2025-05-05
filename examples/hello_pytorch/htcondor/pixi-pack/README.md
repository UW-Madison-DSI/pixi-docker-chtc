# Running the Hello PyTorch example on CHTC GPUs with `pixi-pack`

This example contains the files:

* `hello_pytorch_gpu_pixi-pack.sub`: The HTCondor job submission file
* `hello_pytorch_gpu_pixi-pack.sh`: The execute shells script that contains the commands that are run as the job
* `submit.sh`: The shell script that submits `hello_pytorch_gpu_pixi-pack.sub` to HTCondor as a standard batch job with `condor_submit`
* `interact.sh`: The shell script that submits `hello_pytorch_gpu_pixi-pack.sub` to HTCondor as an interactive job with `condor_submit -interactive`

## Using cached environment archives as an alternative to Linux containers

As the Pixi environment is fully specified through the `pixi.lock` lock file, it is possible to download all of the environment conda packages and export them into an archive with [`pixi-pack`](https://pixi.sh/latest/deployment/pixi_pack/).

```
pixi-pack pack --environment prod --platform linux-64 pixi.toml
```

This will create an `environment.tar` archive with all of the packages required to create the Pixi environment.
Once can use `pixi-pack unpack` to unpack the archive and create the environment locally on another machine.

In the event that the target machine does not have `pixi-pack` installed, `pix-pack` can create a self-extracting binary `environment.sh` with

```
pixi-pack pack --environment prod --platform linux-64 --create-executable pixi.toml
```

which when executed on a target machine of the same platform extracts the environment directory tree and an activation script from the archive

```console
$ chmod +x environment.sh  # depending on permissions
$ ./environment.sh
$ ls
env/
activate.sh
environment.sh
```

These environment archives
As these environment archives are

### Summary

**Advantages**

* Another form of distribution for a fully reproducible software environment.
* No requirement on Linux containers for use.
* No requirement on Pixi for use.
* The time to go from creating a software environment to using it can be reduced to a few minutes.

**Disadvantages**

* As no Linux container is being mounted to the worker node, the HTCondor job requires substantially larger memory and disk resources than a typical job to be able to unpack the archive into the environment directory tree.
* As the software environment needs to get unpacked for each job, this can take a substantial amount of time (multiple minutes) before any code execution can begin.

> [!TIP]
> This is why Linux container images being **executable filesystems** can be so important to their use.
> No unpacking of the filesystem is needed, so Linux containers can start and be used almost immediately.
