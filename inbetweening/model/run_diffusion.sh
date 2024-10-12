#!/bin/bash
#SBATCH --job-name=inbetweening_model
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log 
#SBATCH --time=00:30:00
#SBATCH -N 1

python diffusion.py fit --config ./default_config.yaml

# Script ends here