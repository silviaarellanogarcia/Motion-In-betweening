#!/bin/bash
#SBATCH --gpus 1
#SBATCH --job-name=inbetweening_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log 
#SBATCH --time=12:00:00

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate inbetweening
export PYTHONPATH=$PYTHONPATH:/proj/diffusion-inbetweening
echo $PYTHONPATH
cd model/
python generate_samples.py

# Script ends here