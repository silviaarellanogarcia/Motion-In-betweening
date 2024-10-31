import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.model.diffusion import DiffusionModel
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root
from inbetweening.data_processing.utils import compute_global_positions_in_a_sample
from inbetweening.utils.convert_to_bvh import write_bvh

path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_correct_dims/version_4/checkpoints/epoch=51-step=3172.ckpt'
config_file_corresponding_to_ckpt = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_correct_dims/version_4/config.yaml'

# Load the config file
with open(config_file_corresponding_to_ckpt, 'r') as f:
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
type_model = config['model']['type_model']
kernel_size = config['model']['kernel_size']

model = DiffusionModel.load_from_checkpoint(
    path_to_checkpoint,
    beta_start=beta_start,
    beta_end=beta_end,
    n_diffusion_timesteps=n_diffusion_timesteps,
    lr=lr,
    gap_size=gap_size,
    type_masking=type_masking,
    time_emb_dim=time_emb_dim,
    window=window,
    n_joints=n_joints,
    down_channels=down_channels,
    type_model=type_model,
    kernel_size=kernel_size
)

data_module = Lafan1DataModule(
    data_dir='/proj/diffusion-inbetweening/data',
    batch_size=1,
    window=50,
    offset=20
)

data_module.setup(stage='test')

# Get a single sample from the test dataset
sample_index = 1  # Adjust this index as needed
test_dataset = data_module.test_dataset
sample = test_dataset[sample_index]
sample = {key: value.to(model.device) for key, value in sample.items()}

# ATTENTION! For generating the BVH take into account that the X is local (except the root), and the Q is global.
denoised_X, denoised_Q = model.generate_samples(sample['X'], sample['Q'])

print("NORMALIZED ORIGINAL SAMPLE X: ", sample['X'][22:27,0,:])
print("NORMALIZED DENOISED SAMPLE X: ", denoised_X[22:27,0,:])

# Convert mean_X and std_X from NumPy arrays to PyTorch tensors
mean_X = torch.tensor(test_dataset.training_mean_X, dtype=torch.float32).to(model.device)
std_X = torch.tensor(test_dataset.training_std_X, dtype=torch.float32).to(model.device)

# Undo the normalization on denoised_X (back to the original scale)
denoised_X_original = denoised_X * std_X + mean_X

# Undo the normalization on the sample data as well
sample_X_original = sample['X'] * std_X + mean_X

print('Inbetweening finished!')
print("ORIGINAL SAMPLE X: ", sample_X_original[22:27,0,:])
print("DENOISED SAMPLE X: ", denoised_X_original[22:27,0,:])

print("ORIGINAL SAMPLE Q: ", sample['Q'][22:27,0,:])
print("DENOISED SAMPLE Q: ", denoised_Q[22:27,0,:])

# Plot the 3D skeleton
X_gt_global = compute_global_positions_in_a_sample(sample_X_original, sample['Q'], sample['parents'])
X_denoised_global = compute_global_positions_in_a_sample(denoised_X, denoised_Q, sample['parents'])
# plot_3d_skeleton_with_lines(torch.tensor(X_gt_global).unsqueeze(0), sample['parents'], sequence_index=0, frames_range=(24, 25))
# plot_3d_skeleton_with_lines(torch.tensor(X_denoised_global).unsqueeze(0), sample['parents'], sequence_index=0, frames_range=(24, 25))

# Plot the root's path
# plot_root(sample_X_original.unsqueeze(0)[:, :, 0, :].detach().numpy(), start_frame=0, end_frame=49, sequence_index=0)
# plot_root(denoised_X_original.unsqueeze(0)[:, :, 0, :].detach().numpy(), start_frame=0, end_frame=49, sequence_index=0)

# Generate BVH files
write_bvh('output_original.bvh', X=sample_X_original, Q_global=sample['Q'], parents=sample['parents'])
write_bvh('output_denoised.bvh', X=denoised_X_original, Q_global=denoised_Q, parents=sample['parents'])