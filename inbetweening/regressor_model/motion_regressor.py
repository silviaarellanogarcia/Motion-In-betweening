import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from inbetweening.data_processing.process_data import Lafan1DataModule

class MotionRegressor(pl.LightningModule):
    def __init__(self, input_size: int = 3 + 4, hidden_size: int = 128, lr: float = 0.001, gap_size: int = 2, type_masking: str = 'continued'):
        super(MotionRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_position = nn.Linear(hidden_size, 3)  # Output for position
        self.fc3_rotation = nn.Linear(hidden_size, 4)  # Output for quaternion
        self.learning_rate = lr
        self.gap_size = gap_size
        self.type_masking = type_masking

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        position = self.fc3_position(x)
        rotation = self.fc3_rotation(x)
        return position, rotation
    
    def masking(self, n_frames, gap_size, type='continued'):
        """
        Masks the information of some frames in a motion sequence.
        Type can be 'continued' (the masked frames are one after the other) or 'spread' (the masked frames are placed randdomly on the sequence, except first and last position).
        """
        ## Shape of X: batch_size, frames, joints, position_dims
        ## Shape of Q: batch_size, frames, joints, quaternnion_dims

        masked_frames = []

        if gap_size > n_frames - 2:
            raise AssertionError('Your gap size is bigger or equal than the number of frames in your sequence minus 2.')
        if gap_size/n_frames >= 0.7:
            print('Attention!! You are masking more than the 70% of frames!')

        if type == 'continued':
            start_frame = int((n_frames - gap_size) / 2)
            masked_frames = list(range(start_frame, start_frame + gap_size))
        
        elif type == 'spread':
            # Selection of 'gap_size' frames to mask, excluding the first and last frames
            masked_frames = torch.randperm(n_frames - 2)[:gap_size] + 1  # +1 to exclude frame 0
            masked_frames = masked_frames.tolist()  # Convert to list
            masked_frames = sorted(set(masked_frames))

        return masked_frames

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        X = batch['X']
        Q = batch['Q']

        # Masking
        masked_frames = self.masking(n_frames=X.shape[1], gap_size=self.gap_size, type=self.type_masking)
        # masked_frames_tensor = torch.tensor(masked_frames).view(-1, 1)
        # masked_frames_tensor = masked_frames_tensor.view(-1)
        masked_X = X.clone()
        masked_X[:, masked_frames, :, :] = 0.0
        masked_Q = Q.clone()
        masked_Q[:, masked_frames, :, :] = 0.0

        # Concatenate the input data
        input_combined = torch.cat((masked_X, masked_Q), dim=-1)

        # Forward pass
        predicted_position, predicted_rotation = self(input_combined)

        # Compute loss
        loss_position = nn.MSELoss()(predicted_position[:, masked_frames, 0, :], X[:, masked_frames, 0, :])
        loss_rotation = nn.MSELoss()(predicted_rotation[:, masked_frames, :, :], Q[:, masked_frames, :, :])
        total_loss = loss_position + loss_rotation

        # Log losses
        self.log('train_loss_position', loss_position, prog_bar=True)
        self.log('train_loss_rotation', loss_rotation, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        """Define the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    print("AM I USING GPU? ", torch.cuda.is_available())
    logger_config = {
        'class_path': 'pytorch_lightning.loggers.TensorBoardLogger',
        'init_args': {                      # Use init_args instead of params
            'save_dir': 'lightning_logs',
            'name': 'regressor_1',
            'version': None
        }
    }

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,  # Keep 10 checkpoints
        monitor='validation_total_loss',
        mode="min"
    )

    # Use LightningCLI with the updated logger configuration
    LightningCLI(MotionRegressor, 
                 Lafan1DataModule, 
                 trainer_defaults={
                     'logger': logger_config,
                     'callbacks': [checkpoint_callback],
                    #  'overfit_batches': 1 ## TODO: AT SOME POINT REMOVE THE OVERFITTING
    })

    ## COMMAND: python motion_regressor.py fit --config ./config_regressor.yaml
    ## For continue training from a checkpoint: python motion_regressor.py fit --config ./config_regressor.yaml --ckpt_path PATH
