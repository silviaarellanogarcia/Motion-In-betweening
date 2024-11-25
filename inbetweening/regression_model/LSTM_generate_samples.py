import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule
from inbetweening.evaluation.baseline import full_interpolation
from inbetweening.model.diffusion import DiffusionModel
from inbetweening.regression_model.LSTM_model import MotionLSTM
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root
from inbetweening.data_processing.utils import compute_global_positions_in_a_sample
from inbetweening.utils.convert_to_bvh import write_bvh
import pymotion.rotations.ortho6d as sixd

samples_to_generate = 51
gap_sizes = [5, 10, 15]
save_folder_name = 'generated_samples_LSTM'
path_to_checkpoint = '/proj/diffusion-inbetweening/inbetweening/regression_model/lightning_logs/motionLSTM/version_12/checkpoints/epoch=12450-step=759511.ckpt'
config_file_corresponding_to_ckpt = '/proj/diffusion-inbetweening/inbetweening/regression_model/lightning_logs/motionLSTM/version_12/config.yaml'

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

data_module = Lafan1DataModule(
    data_dir='/proj/diffusion-inbetweening/data',
    batch_size=1,
    window=window,
    offset=offset,
)

data_module.setup(stage='test')
test_dataset = data_module.test_dataset

for gap_size in gap_sizes:
    print(f"GAP: {gap_size}")

    model = MotionLSTM.load_from_checkpoint(
        path_to_checkpoint,
        hidden_size=hidden_size,
        lr=lr,
        gap_size=gap_size,
        type_masking=type_masking,
        n_frames=window,
        step_threshold=step_threshold, 
        max_gap_size=max_gap_size,
        n_layers=n_layers
    )

    for i in range(samples_to_generate):
        print(f"\t ITERATION: {i}, SAMPLE: {i*50}")

        # Get a single sample from the test dataset
        sample_index = i * 50  # Adjust this index as needed
        sample = test_dataset[sample_index]
        sample = {key: value.to(model.device) for key, value in sample.items()}

        # ATTENTION! The Q is global.
        predicted_Q, masked_frames = model.generate_samples(sample['X'], sample['Q'], model)

        print(masked_frames)

        print('Inbetweening finished!')

        # print("ORIGINAL SAMPLE Q: ", sample['Q'][masked_frames,0,:])
        # print("DENOISED SAMPLE Q: ", predicted_Q[masked_frames,0,:])

        # Pass from Ortho6D to quaternions
        original_Q_quat = sample['Q'].reshape(sample['Q'].shape[0], sample['Q'].shape[1], 3, 2) # Shape (frames, joints, 6) --> Shape (frames, joints, 3, 2)
        original_Q_quat = sixd.to_quat(original_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

        predicted_Q_quat = predicted_Q.reshape(predicted_Q.shape[0], predicted_Q.shape[1], 3, 2)
        predicted_Q_quat = sixd.to_quat(predicted_Q_quat.cpu().numpy())  # Convert to NumPy for sixd

        # Convert back to PyTorch tensors
        original_Q_quat = torch.tensor(original_Q_quat, device=sample['X'].device)
        predicted_Q_quat = torch.tensor(predicted_Q_quat, device=sample['X'].device)

        # Generate BVH files
        if gap_size == gap_sizes[0]:
            write_bvh(f'./{save_folder_name}/LSTM_output_{sample_index}_original.bvh', X=sample['X'], Q_global=original_Q_quat, parents=sample['parents'])

        write_bvh(f'./{save_folder_name}/LSTM_output_{sample_index}_generated_{gap_size}_fr.bvh', X=sample['X'], Q_global=predicted_Q_quat, parents=sample['parents'])