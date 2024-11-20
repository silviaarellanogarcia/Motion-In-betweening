import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.evaluation.baseline import full_interpolation
from inbetweening.model.unet import SimpleUnet
from inbetweening.regression_model.UNet_model import UNetModel
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root
from inbetweening.data_processing.utils import compute_global_positions_in_a_sample
from inbetweening.utils.convert_to_bvh import write_bvh
import pymotion.rotations.ortho6d as sixd

#path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_11/checkpoints/epoch=22779-step=1389580.ckpt'
path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_12/checkpoints/epoch=28119-step=1715320.ckpt'
config_file_corresponding_to_ckpt = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_12/config.yaml'

# Load the config file
with open(config_file_corresponding_to_ckpt, 'r') as f:
    config = yaml.safe_load(f)

# Prepare all the hyperparameters that are necessary for the model
lr = config['model']['lr']
gap_size = 21
type_masking = config['model']['type_masking']
time_emb_dim = config['model']['time_emb_dim']
window = config['model']['window']
n_joints = config['model']['n_joints']
down_channels = config['model']['down_channels']
type_model = config['model']['type_model']
kernel_size = config['model']['kernel_size']
offset = config['data']['offset']
step_threshold = config['model']['step_threshold']
max_gap_size = config['model']['max_gap_size']

model = UNetModel.load_from_checkpoint(
    path_to_checkpoint,
    lr=lr,
    gap_size=gap_size,
    type_masking=type_masking,
    time_emb_dim=time_emb_dim,
    window=window,
    n_joints=n_joints,
    down_channels=down_channels,
    type_model=type_model,
    kernel_size=kernel_size,
    step_threshold=step_threshold, 
    max_gap_size=max_gap_size
)

data_module = Lafan1DataModule(
    data_dir='/proj/diffusion-inbetweening/data',
    batch_size=1,
    window=window,
    offset=offset,
)

data_module.setup(stage='test')

# Get a single sample from the test dataset
sample_index = 2500  # Adjust this index as needed
test_dataset = data_module.test_dataset
sample = test_dataset[sample_index]
sample = {key: value.to(model.device) for key, value in sample.items()}

# ATTENTION! The Q is global.
model.eval()
with torch.no_grad():
    X_0 = sample['X'].unsqueeze(0)
    Q_0 = sample['Q'].unsqueeze(0) ## This adds the batch dimension

    # Masking
    masked_frames = model.masking(n_frames=Q_0.shape[1], gap_size=gap_size, type=type_masking)
    masked_Q = Q_0.clone()
    masked_Q[:, masked_frames, :, :] = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check GPU availability
    t = torch.zeros(Q_0.shape[0]).to(device)
    Q_pred = SimpleUnet(X_0, masked_Q) # The timestep could be removed, since we are generating all at once

    masked_Q[:, masked_frames, :, :] = Q_pred[:, masked_frames, :, :].float()

predicted_Q, masked_frames = masked_Q[0], masked_frames


print(masked_frames)

print('Inbetweening finished!')

print("ORIGINAL SAMPLE Q: ", sample['Q'][masked_frames,0,:])
print("DENOISED SAMPLE Q: ", predicted_Q[masked_frames,0,:])

# Pass from Ortho6D to quaternions
original_Q_quat = sample['Q'].reshape(sample['Q'].shape[0], sample['Q'].shape[1], 3, 2) # Shape (frames, joints, 6) --> Shape (frames, joints, 3, 2)
original_Q_quat = sixd.to_quat(original_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

predicted_Q_quat = predicted_Q.reshape(predicted_Q.shape[0], predicted_Q.shape[1], 3, 2)
predicted_Q_quat = sixd.to_quat(predicted_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

# Convert back to PyTorch tensors
original_Q_quat = torch.tensor(original_Q_quat, device=sample['X'].device)
predicted_Q_quat = torch.tensor(predicted_Q_quat, device=sample['X'].device)


# Generate BVH files
write_bvh('UNET_original.bvh', X=sample['X'], Q_global=original_Q_quat, parents=sample['parents'])
write_bvh('UNET_denoised_21_fr.bvh', X=sample['X'], Q_global=predicted_Q_quat, parents=sample['parents'])