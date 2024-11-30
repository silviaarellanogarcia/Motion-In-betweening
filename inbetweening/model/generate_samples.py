import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize
import yaml
import tqdm

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.evaluation.baseline import full_interpolation
from inbetweening.model.diffusion import DiffusionModel
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root
from inbetweening.data_processing.utils import compute_global_positions_in_a_sample
from inbetweening.utils.convert_to_bvh import write_bvh
import pymotion.rotations.ortho6d as sixd

gap_sizes = [5]
save_folder_name = 'generated_samples_fr5_v11'
# path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_9/checkpoints/epoch=1579-step=96380.ckpt'
path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_11/checkpoints/epoch=22779-step=1389580.ckpt'
# path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X_kernel7/version_0/checkpoints/epoch=18199-step=1110200.ckpt'
config_file_corresponding_to_ckpt = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_Q_and_X/version_9/config.yaml'

# Load the config file
with open(config_file_corresponding_to_ckpt, 'r') as f:
    config = yaml.safe_load(f)

# Prepare all the hyperparameters that are necessary for the model
beta_start = config['model']['beta_start']
beta_end = config['model']['beta_end']
n_diffusion_timesteps = config['model']['n_diffusion_timesteps']
lr = config['model']['lr']
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

batch_size = 256

data_module = Lafan1DataModule(
        data_dir='/proj/diffusion-inbetweening/data',
        batch_size=batch_size,
        window=window,
        offset=offset,
    )

data_module.setup(stage='validate')
test_dataset = data_module.val_dataset ## TODO: CHANGE TO TEST DATASET!!!
test_dataloader = data_module.val_dataloader() ## TODO: CHANGE TO TEST DATALOADER!!!
samples_to_generate = len(test_dataset)//3 ## TODO: REMOVE THE 3!!!
print("Samples to generate: ", samples_to_generate)

for gap_size in gap_sizes:
    print(f"GAP: {gap_size}")

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
        kernel_size=kernel_size,
        step_threshold=step_threshold, 
        max_gap_size=max_gap_size
    )

    iterator = tqdm.tqdm(iter(test_dataloader), total=len(test_dataset)//batch_size)
    for batch_index, sample in enumerate(iterator):
        # Get a single sample from the test dataset
        sample = {key: value.to(model.device) for key, value in sample.items()}

        # ATTENTION! The Q is global.
        denoised_Q, masked_frames = model.generate_samples(sample['X'], sample['Q'])

        # Pass from Ortho6D to quaternions
        original_Q_quat = sample['Q'].reshape(sample['Q'].shape[0], sample['Q'].shape[1], sample['Q'].shape[2], 3, 2) # Shape (frames, joints, 6) --> Shape (frames, joints, 3, 2)
        original_Q_quat = sixd.to_quat(original_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

        denoised_Q_quat = denoised_Q.reshape(denoised_Q.shape[0], denoised_Q.shape[1], denoised_Q.shape[2], 3, 2)
        denoised_Q_quat = sixd.to_quat(denoised_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

        # Convert back to PyTorch tensors
        original_Q_quat = torch.tensor(original_Q_quat, device=sample['X'].device)
        denoised_Q_quat = torch.tensor(denoised_Q_quat, device=sample['X'].device)

        # Generate BVH files
        # if gap_size == gap_sizes[0]:
            # write_bvh(f'./{save_folder_name}/diff_output_{batch_index * batch_size + sample_index}_original_{gap_size}_fr.bvh', X=sample['X'][sample_index], Q_global=original_Q_quat, parents=sample['parents'][0])
        for sample_index, Q_sample in enumerate(denoised_Q_quat):
            try:
                write_bvh(f'./{save_folder_name}/diff_output_{batch_index * batch_size + sample_index}_denoised_{gap_size}_fr.bvh', X=sample['X'][sample_index], Q_global=Q_sample, parents=sample['parents'][0])
            except:
                continue