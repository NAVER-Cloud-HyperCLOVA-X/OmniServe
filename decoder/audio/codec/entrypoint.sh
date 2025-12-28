#!/bin/bash

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
exec torchserve $@