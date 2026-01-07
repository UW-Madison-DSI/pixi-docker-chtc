#!/usr/bin/env bash

condor_submit ./mpi_ring.sub

sleep 1

condor_q "${USER}"
