import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.model.diffusion import DiffusionModel
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root
from inbetweening.data_processing.utils import compute_global_positions_in_a_sample

# Load the config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Prepare all the hyperparameters that are necessary for the model
beta_start = config['model']['beta_start']
beta_end = config['model']['beta_end']
n_diffusion_timesteps = config['model']['n_diffusion_timesteps']
lr = config['model']['lr']
gap_size = config['model']['gap_size']
type_masking = config['model']['type_masking']
time_emb_dim = config['model']['time_emb_dim']
window = config['model']['window']
n_joints = config['model']['n_joints']
down_channels = config['model']['down_channels']

model = DiffusionModel.load_from_checkpoint(
    '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_init/version_17/checkpoints/epoch=171-step=41452.ckpt',
    beta_start=beta_start,
    beta_end=beta_end,
    n_diffusion_timesteps=n_diffusion_timesteps,
    lr=lr,
    gap_size=gap_size,  # Use default value if not specified
    type_masking=type_masking,  # Use default value if not specified
    time_emb_dim=time_emb_dim,  # Use default value if not specified
    window=window,  # Use default value if not specified
    n_joints=n_joints,  # Use default value if not specified
    down_channels=down_channels # Use default value if not specified
)

data_module = Lafan1DataModule(
    data_dir='/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data1',
    batch_size=1,
    window=50,
    offset=20
)

data_module.setup(stage='test')

# Get a single sample from the test dataset
sample_index = 0  # Adjust this index as needed
test_dataset = data_module.test_dataset
sample = test_dataset[sample_index]
sample = {key: value.to(model.device) for key, value in sample.items()}

# ATTENTION! For generating the BVH take into account that the X is local (except the root), and the Q is global.
denoised_X, denoised_Q = model.generate_samples(sample['X'], sample['Q'])
print('Inbetweening finished!')
print("ORIGINAL SAMPLE: ", denoised_X[22:27,0,:])
print("DENOISED SAMPLE: ", sample['X'][22:27,0,:])

X_gt_global = compute_global_positions_in_a_sample(sample['X'], sample['Q'], sample['parents'])
X_denoised_global = compute_global_positions_in_a_sample(denoised_X, denoised_Q, sample['parents'])

plot_3d_skeleton_with_lines(torch.tensor(X_gt_global).unsqueeze(0), sample['parents'], sequence_index=0, frames_range=(24, 25))
plot_3d_skeleton_with_lines(torch.tensor(X_denoised_global).unsqueeze(0), sample['parents'], sequence_index=0, frames_range=(24, 25))

# plot_root(sample['X'].unsqueeze(0)[:, :, 0, :].detach().numpy(), start_frame=0, end_frame=49, sequence_index=0)
# plot_root(denoised_X.unsqueeze(0)[:, :, 0, :].detach().numpy(), start_frame=0, end_frame=49, sequence_index=0)