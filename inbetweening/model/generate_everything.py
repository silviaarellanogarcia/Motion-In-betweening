import torch
import tqdm
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.model.diffusion_without_X import DiffusionModel
from inbetweening.position_model.LSTM_model_positions import PositionLSTM
from inbetweening.utils.convert_to_bvh import write_bvh
import pymotion.rotations.ortho6d as sixd

gap_sizes = [5, 10, 15]
save_folder_name = 'generated_all_Xv27_Qv1'

path_to_checkpoint_pos = '/proj/diffusion-inbetweening/inbetweening/position_model/lightning_logs/positionLSTM/version_27/checkpoints/epoch=8679-step=529480.ckpt'
config_file_corresponding_to_ckpt_pos = '/proj/diffusion-inbetweening/inbetweening/position_model/lightning_logs/positionLSTM/version_27/config.yaml'

path_to_checkpoint_rot = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_only_Q/version_1/checkpoints/epoch=15879-step=968680.ckpt'
config_file_corresponding_to_ckpt_rot = '/proj/diffusion-inbetweening/inbetweening/model/lightning_logs/my_model_only_Q/version_1/config.yaml'

# Load the config file
with open(config_file_corresponding_to_ckpt_rot, 'r') as f:
    config_rot = yaml.safe_load(f)

# Load the config file
with open(config_file_corresponding_to_ckpt_pos, 'r') as f:
    config = yaml.safe_load(f)

# Prepare all the hyperparameters that are necessary for the model
beta_start = config_rot['model']['beta_start']
beta_end = config_rot['model']['beta_end']
n_diffusion_timesteps = config_rot['model']['n_diffusion_timesteps']
lr_rot = config_rot['model']['lr']
type_masking = config_rot['model']['type_masking']
time_emb_dim = config_rot['model']['time_emb_dim']
window = config_rot['model']['window']
n_joints = config_rot['model']['n_joints']
down_channels = config_rot['model']['down_channels']
type_model = config_rot['model']['type_model']
kernel_size = config_rot['model']['kernel_size']
offset = config_rot['data']['offset']
step_threshold_rot = config_rot['model']['step_threshold']
max_gap_size = config_rot['model']['max_gap_size']

lr_pos = config['model']['lr']
step_threshold_pos = config['model']['step_threshold']
hidden_size = config['model']['hidden_size']
n_layers = config['model']['n_layers']
lower_limit_gap = config['model']['lower_limit_gap']
upper_limit_gap = config['model']['upper_limit_gap']

batch_size = 256

data_module = Lafan1DataModule(
        data_dir='/proj/diffusion-inbetweening/data',
        batch_size=batch_size,
        window=window,
        offset=offset,
    )

data_module.setup(stage='test')
test_dataset = data_module.test_dataset

data_module.setup(stage='test')
test_dataset = data_module.test_dataset
test_dataloader = data_module.test_dataloader()
samples_to_generate = len(test_dataset)
print("Samples to generate: ", samples_to_generate)

for gap_size in gap_sizes:
    print(f"GAP: {gap_size}")

    model_rot = DiffusionModel.load_from_checkpoint(
        path_to_checkpoint_rot,
        beta_start=beta_start,
        beta_end=beta_end,
        n_diffusion_timesteps=n_diffusion_timesteps,
        lr=lr_rot,
        gap_size=gap_size,
        type_masking=type_masking,
        time_emb_dim=time_emb_dim,
        window=window,
        n_joints=n_joints,
        down_channels=down_channels,
        type_model=type_model,
        kernel_size=kernel_size,
        step_threshold=step_threshold_rot, 
        max_gap_size=max_gap_size
    )

    model_pos = PositionLSTM.load_from_checkpoint(
        path_to_checkpoint_pos,
        hidden_size=hidden_size,
        lr=lr_pos,
        gap_size=gap_size,
        type_masking=type_masking,
        n_frames=window,
        step_threshold=step_threshold_pos, 
        max_gap_size=max_gap_size,
        n_layers=n_layers,
        lower_limit_gap=lower_limit_gap,
        upper_limit_gap=upper_limit_gap
    )

    iterator = tqdm.tqdm(iter(test_dataloader), total=len(test_dataset)//batch_size)
    for batch_index, sample in enumerate(iterator):
        # Get a single sample from the test dataset
        sample = {key: value.to(model_rot.device) for key, value in sample.items()}

        # ATTENTION! The Q is global.
        denoised_Q, masked_frames = model_rot.generate_samples(sample['Q'])
        predicted_X, masked_frames = model_pos.generate_samples(sample['X'], model_pos)

        # Pass from Ortho6D to quaternions
        original_Q_quat = sample['Q'].reshape(sample['Q'].shape[0], sample['Q'].shape[1], sample['Q'].shape[2], 3, 2) # Shape (frames, joints, 6) --> Shape (frames, joints, 3, 2)
        original_Q_quat = sixd.to_quat(original_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

        denoised_Q_quat = denoised_Q.reshape(denoised_Q.shape[0], denoised_Q.shape[1], denoised_Q.shape[2], 3, 2)
        denoised_Q_quat = sixd.to_quat(denoised_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

        # Convert back to PyTorch tensors
        original_Q_quat = torch.tensor(original_Q_quat, device=sample['X'].device)
        denoised_Q_quat = torch.tensor(denoised_Q_quat, device=sample['X'].device)

        # Generate BVH files
        for sample_index, Q_sample in enumerate(denoised_Q_quat):
            try:
                write_bvh(f'./{save_folder_name}/diff_output_{batch_index * batch_size + sample_index}_denoised_{gap_size}_fr.bvh', X=predicted_X[sample_index], Q_global=Q_sample, parents=sample['parents'][0])

                if gap_size == gap_sizes[0]:
                    write_bvh(f'./{save_folder_name}/diff_output_{batch_index * batch_size + sample_index}_original.bvh', X=sample['X'][sample_index], Q_global=original_Q_quat[sample_index], parents=sample['parents'][0])
            except:
                continue