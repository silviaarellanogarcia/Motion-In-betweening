import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from inbetweening.data_processing.process_data import Lafan1DataModule

class StopTrainingOnGapSize(pl.Callback):
    def __init__(self, gap_size_threshold=30):
        self.gap_size_threshold = gap_size_threshold

    def on_epoch_start(self, trainer, pl_module):
        """Stop the training if gap_size is equal to or greater than the threshold."""
        if pl_module.gap_size == self.gap_size_threshold:
            print(f"Stopping training because gap_size is {self.gap_size_threshold}.")
            trainer.should_stop = True  # This stops the training process

class MotionLSTM(pl.LightningModule):
    def __init__(self, hidden_size: int = 128, lr: float = 0.001, gap_size: int = 1, type_masking: str = 'continued', n_frames: int = 50, step_threshold: int = 8000, max_gap_size: int = 15, n_layers: int=2):
        super(MotionLSTM, self).__init__()
        self.learning_rate = lr
        self.gap_size = gap_size
        self.max_gap_size = max_gap_size
        self.type_masking = type_masking
        self.n_frames = n_frames
        self.steps_since_last_gap_increase = 0
        self.step_threshold = step_threshold

        # Temporal processing (Shared across root positions and joint rotations)
        self.temporal_processing = nn.LSTM(input_size=n_frames, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

        # Final prediction layer (for each joint's 6D rotation in the masked frames)
        self.output_layer = nn.Linear(hidden_size, self.n_frames)


    def forward(self, X, Q):
        """Forward pass through the network."""

        batch_size, frames, joints, pos_dims = X.shape
        X = X.view(batch_size, frames, joints * pos_dims)  # Flatten positions (X)

        batch_size, frames, joints, angle_dims = Q.shape
        Q = Q.view(batch_size, frames, joints * angle_dims)  # Flatten Ortho6D angles (Q)

        # Concatenate positions (X) and Ortho6D angles (Q) along the joint dimension
        X_and_Q = torch.cat((X, Q), dim=2)  # X and Q are concatenated along the channels axis
        X_and_Q = torch.permute(X_and_Q, (0, 2, 1))  # Change to (batch_size, channels, frames)

        # Feed combined features into LSTM for temporal processing
        lstm_out, _ = self.temporal_processing(X_and_Q)

        # Output layer for predicting quaternions for each joint
        output = self.output_layer(lstm_out)  # (batch_size, num_joints*9, frames)
        output = output.permute(0, 2, 1) # (batch_size, frames, num_joints*9)
        output = output.view(batch_size, frames, joints, 9)

        return output[:, :, :, 3:]

    
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
        if self.steps_since_last_gap_increase >= self.step_threshold:  # TODO: Change 5 with a parameter indicating the maximum gap
            self.gap_size = min(self.gap_size + 1, self.max_gap_size)  # TODO: Adjust maximum gap parameter
            self.step_threshold += 3000
            self.steps_since_last_gap_increase = 0

        X = batch['X']
        Q = batch['Q']

        # Masking
        masked_frames = self.masking(n_frames=Q.shape[1], gap_size=self.gap_size, type=self.type_masking)

        masked_Q = Q.clone()
        masked_Q[:, masked_frames, :, :] = 0.0

        # Forward pass
        predicted_rotation = self(X, masked_Q)

        # Compute loss
        loss_rotation = nn.MSELoss()(predicted_rotation[:, masked_frames, :, :], Q[:, masked_frames, :, :])

        # Log losses
        self.log('train_loss', loss_rotation, prog_bar=True)
        self.log('gap_size', self.gap_size, prog_bar=False)

        self.steps_since_last_gap_increase += 1

        return loss_rotation
    
    def validation_step(self, batch, batch_idx):
        """Perform a training step."""
        X = batch['X']
        Q = batch['Q']

        # Masking
        masked_frames = self.masking(n_frames=Q.shape[1], gap_size=self.gap_size, type=self.type_masking)

        masked_Q = Q.clone()
        masked_Q[:, masked_frames, :, :] = 0.0

        # Forward pass
        predicted_rotation = self(X, masked_Q)

        # Compute loss
        loss_rotation = nn.MSELoss()(predicted_rotation[:, masked_frames, :, :], Q[:, masked_frames, :, :])

        # Log losses
        self.log('validation_loss', loss_rotation, prog_bar=True)

        return loss_rotation

    def configure_optimizers(self):
        """Define the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def generate_samples(self, X_0, Q_0, model):
        self.eval()
        with torch.no_grad():
            X_0 = X_0.unsqueeze(0)
            Q_0 = Q_0.unsqueeze(0) ## This adds the batch dimension

            masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

            masked_Q = Q_0.clone()
            masked_Q[:, masked_frames, :, :] = 0.0

            # Calculate loss
            Q_pred = model(X_0, masked_Q) ### I will have X and Q together and I have to separate them.
            masked_Q[:, masked_frames, :, :] = Q_pred[:, masked_frames, :, :].float()

        return masked_Q[0], masked_frames


if __name__ == '__main__':
    print("AM I USING GPU? ", torch.cuda.is_available())
    logger_config = {
        'class_path': 'pytorch_lightning.loggers.TensorBoardLogger',
        'init_args': {                      # Use init_args instead of params
            'save_dir': 'lightning_logs',
            'name': 'motionLSTM',
            'version': None
        }
    }

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,  # Keep 10 checkpoints
        monitor='train_loss',
        mode="min"
    )

    stop_training_callback = StopTrainingOnGapSize(gap_size_threshold=30)

    # Use LightningCLI with the updated logger configuration
    LightningCLI(MotionLSTM, 
                 Lafan1DataModule, 
                 trainer_defaults={
                     'logger': logger_config,
                     'callbacks': [checkpoint_callback, stop_training_callback],
                    #  'overfit_batches': 1 ## TODO: AT SOME POINT REMOVE THE OVERFITTING
    })

    ## COMMAND: python LSTM_model.py fit --config ./LSTM_config.yaml
    ## For continue training from a checkpoint: python motion_regressor.py fit --config ./config_regressor.yaml --ckpt_path PATH