#!/usr/bin/env bash

# detailed logging to stderr
set -x

# HTCondor sets these environment variables for parallel jobs:
# _CONDOR_PROCNO - rank of this process
# _CONDOR_NPROCS - total number of processes
# _CONDOR_SCRATCH_DIR - scratch directory

echo -e "# Installing Pixi"
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bashrc
echo -e "# Installing environment"
pixi install
sleep 10

echo "Starting MPI wrapper on node: $(hostname)"
echo "Process rank: ${_CONDOR_PROCNO}"
echo "Total processes: ${_CONDOR_NPROCS}"

# HTCondor creates a machine file with all allocated nodes
MACHINE_FILE="${_CONDOR_SCRATCH_DIR}/.condor_machine_list"

echo "Machine file contents:"
cat "${MACHINE_FILE}"

# Launch the MPI program across all nodes
# HTCondor's parallel universe handles the distribution
pixi run mpirun -np "${_CONDOR_NPROCS}" \
    -hostfile "${MACHINE_FILE}" \
    python ./mpi_ring.py

exit_code=$?
echo "MPI job completed with exit code: ${exit_code}"
exit "${exit_code}"
