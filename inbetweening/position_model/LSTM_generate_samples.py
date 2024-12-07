import torch
import tqdm
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.position_model.LSTM_model_positions import PositionLSTM
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root, plot_root_with_real
from inbetweening.data_processing.utils import compute_global_positions_in_a_sample
from inbetweening.utils.convert_to_bvh import write_bvh
import pymotion.rotations.ortho6d as sixd

gap_sizes = [5, 10, 15]
save_folder_name = 'root_plot_and_motion'

path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/position_model/lightning_logs/positionLSTM/version_27/checkpoints/epoch=8679-step=529480.ckpt'
config_file_corresponding_to_ckpt = '/proj/diffusion-inbetweening/inbetweening/position_model/lightning_logs/positionLSTM/version_27/config.yaml'

# Load the config file
with open(config_file_corresponding_to_ckpt, 'r') as f:
    config = yaml.safe_load(f)

# Prepare all the hyperparameters that are necessary for the model
lr = config['model']['lr']
type_masking = config['model']['type_masking']
window = config['data']['window']  # This corresponds to the 'window' value in the config
offset = config['data']['offset']
step_threshold = config['model']['step_threshold']
max_gap_size = config['model']['max_gap_size']
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
test_dataset = data_module.test_dataset ## TODO: CHANGE TO TEST DATASET!!!
test_dataloader = data_module.test_dataloader() ## TODO: CHANGE TO TEST DATALOADER!!!
samples_to_generate = len(test_dataset)//3 ## TODO: REMOVE THE 3!!!
print("Samples to generate: ", samples_to_generate)

for gap_size in gap_sizes:
    print(f"GAP: {gap_size}")

    model = PositionLSTM.load_from_checkpoint(
        path_to_checkpoint,
        hidden_size=hidden_size,
        lr=lr,
        gap_size=gap_size,
        type_masking=type_masking,
        n_frames=window,
        step_threshold=step_threshold, 
        max_gap_size=max_gap_size,
        n_layers=n_layers,
        lower_limit_gap=lower_limit_gap,
        upper_limit_gap=upper_limit_gap
    )

    iterator = tqdm.tqdm(iter(test_dataloader), total=len(test_dataset)//batch_size)
    for batch_index, sample in enumerate(iterator):
        # Get a single sample from the test dataset
        sample = {key: value.to(model.device) for key, value in sample.items()}

        predicted_X, masked_frames = model.generate_samples(sample['X'], model) ## TODO: Corregir la shape

        # Pass from Ortho6D to quaternions
        original_Q_quat = sample['Q'].reshape(sample['Q'].shape[0], sample['Q'].shape[1], sample['Q'].shape[2], 3, 2) # Shape (frames, joints, 6) --> Shape (frames, joints, 3, 2)
        original_Q_quat = sixd.to_quat(original_Q_quat.cpu().numpy())  # Convert to NumPy for sixd
        # Convert back to PyTorch tensors
        original_Q_quat = torch.tensor(original_Q_quat, device=sample['X'].device)

        # Generate plots for the root trajectory
        plot_root_with_real(predicted_X[:, :, 0, :].cpu(), sample['X'][:, :, 0, :].cpu(), start_frame=0, end_frame=49, gap_size=gap_size, output_dir=save_folder_name)

        # Generate BVH files
        for sample_index, pred_X in enumerate(predicted_X):
            try:
                write_bvh(f'./{save_folder_name}/LSTM_output_{batch_index * batch_size + sample_index}_generated_{gap_size}_fr.bvh', X=pred_X, Q_global=original_Q_quat[sample_index], parents=sample['parents'][0])

                if gap_size == gap_sizes[0]:
                    write_bvh(f'./{save_folder_name}/LSTM_output_{batch_index * batch_size + sample_index}_original.bvh', X=sample['X'][sample_index], Q_global=original_Q_quat[sample_index], parents=sample['parents'][0])
            except:
                continue