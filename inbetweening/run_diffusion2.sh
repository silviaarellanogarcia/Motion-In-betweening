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
python diffusion.py fit --config ./config.yaml --ckpt_path /proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_9/checkpoints/epoch=8979-step=547780.ckpt

# Script ends here