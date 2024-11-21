import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule, Lafan1Dataset
from inbetweening.model.mlp import SimpleMLP
from inbetweening.model.unet import SimpleUnet
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

# from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines


class UNetModel(pl.LightningModule):
    def __init__(self, lr: float, gap_size: int, type_masking: str, time_emb_dim: int, window: int, n_joints: int, down_channels: list[int], type_model: str, kernel_size: int, step_threshold: int, max_gap_size: int):
        super().__init__() # Initialize the parent's class before initializing any child

        self.type_model = type_model
        
        self.kernel_size = kernel_size

        self.model = SimpleUnet(time_emb_dim, window, n_joints, down_channels, kernel_size)
        self.lr = lr
        self.window = window
        self.n_joints = n_joints

        self.gap_size = gap_size
        self.type_masking = type_masking
        self.step_threshold = step_threshold
        self.max_gap_size = max_gap_size
        self.steps_since_last_gap_increase = 0  # Counter for steps since last gap increase
    
    def masking(self, n_frames, gap_size, type='continued'):
        """
        Masks the information of some frames in a motion sequence.
        Type can be 'continued' (the masked frames are one after the other) or 'spread' (the masked frames are placed randdomly on the sequence, except first and last position).
        """
        ## Shape of X: batch_size, frames, joints, position_dims
        ## Shape of Q: batch_size, frames, joints, angle_dims

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

        return masked_frames
    
    
    def get_loss(self, model, X_0, Q_0, masked_frames):
        """
        Compute the loss between the predicted noise and the actual noise for both positions (X_0) and quaternions (Q_0).
        """
        masked_Q = Q_0.clone()
        masked_Q[:, masked_frames, :, :] = 0.0
        
        # Predict noise for both positional and quaternion data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check GPU availability
        t = torch.zeros(Q_0.shape[0]).to(device)
        Q_pred = model(X_0, masked_Q, timestep=t) # The timestep could be deleted now, since we are generating all at once
        
        # Create a tensor for the masked frames
        masked_frames_tensor = torch.tensor(masked_frames, device=device)

        # Calculate the loss
        batch_size, frames, joints, angle_dims = Q_0.shape
        Q_0 = Q_0.view(batch_size, frames, joints * angle_dims)
        Q_0 = torch.permute(Q_0, (0,2,1))

        loss_Q = F.mse_loss(Q_0[:, :, masked_frames_tensor], Q_pred[:, :, masked_frames_tensor], reduction='sum')
        
        return loss_Q
    
    def training_step(self, batch, batch_idx):
        if self.steps_since_last_gap_increase >= self.step_threshold:  # TODO: Change 5 with a parameter indicating the maximum gap
            self.gap_size = min(self.gap_size + 1, self.max_gap_size)  # TODO: Adjust maximum gap parameter
            self.step_threshold += 5000
            self.steps_since_last_gap_increase = 0

        X_0 = batch['X']
        Q_0 = batch['Q']

        masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

        # Calculate loss
        loss_Q = self.get_loss(self.model, X_0, Q_0, masked_frames)
        total_loss = loss_Q / Q_0.shape[0]
        
        # Log both step and epoch loss
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True)

        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True, on_step=True)
        self.log('gap_size', self.gap_size, prog_bar=True, on_step=True)

        # Update the step counter
        self.steps_since_last_gap_increase += 1

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Batch processing
        X_0 = batch['X']
        Q_0 = batch['Q']

        # Masking
        masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

        # Calculate loss
        loss_Q = self.get_loss(self.model, X_0, Q_0, masked_frames)
        total_loss = (loss_Q) / Q_0.shape[0]
        
        # Log loss
        self.log('validation_total_loss', total_loss, prog_bar=True, on_step=True) # We divide the loss by the batch size

        return total_loss
    
    def generate_samples(self, X_0, Q_0):
        self.eval()
        with torch.no_grad():
            X_0 = X_0.unsqueeze(0)
            Q_0 = Q_0.unsqueeze(0) ## This adds the batch dimension

            # Masking
            masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)
            masked_Q = Q_0.clone()
            masked_Q[:, masked_frames, :, :] = 0.0

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            t = torch.zeros(Q_0.shape[0]).to(device)
            Q_pred = self.model(X_0, masked_Q, t) # The timestep could be removed, since we are generating all at once

            batch_size, joints_and_angles, frames = Q_pred.shape
            Q_pred = torch.permute(Q_pred, (0,2,1))
            Q_pred = Q_pred.view(batch_size, frames, self.n_joints, 6)

            masked_Q[:, masked_frames, :, :] = Q_pred[:, masked_frames, :, :].float()

        return masked_Q[0], masked_frames


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9999)

        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer, 
        #         mode='min',
        #         factor=0.3, # Reduce the learning rate by 30%
        #         patience=10, # Number of epochs to wait before reducing LR
        #         min_lr=1e-6, # Minimum learning rate to avoid reducing too much
        #         verbose=True
        #     ),
        #     'monitor': 'train_total_loss', # The metric to monitor; ensure 'val_loss' is logged in validation_step
        #     'interval': 'epoch',   # Check at the end of every epoch
        #     'frequency': 1         # Check after every epoch
        # }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    print("AM I USING GPU? ", torch.cuda.is_available())
    logger_config = {
        'class_path': 'pytorch_lightning.loggers.TensorBoardLogger',
        'init_args': {                      # Use init_args instead of params
            'save_dir': 'lightning_logs',
            'name': 'only_UNet',
            'version': None
        }
    }

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,  # Keep all checkpoints
        monitor='train_total_loss',
        mode="min",
        every_n_epochs=20 # Save every 10 epochs
    )

    # Use LightningCLI with the updated logger configuration
    LightningCLI(UNetModel, 
                 Lafan1DataModule, 
                 trainer_defaults={
                     'logger': logger_config,
                     'callbacks': [checkpoint_callback],
                    #  'overfit_batches': 1 ## TODO: AT SOME POINT REMOVE THE OVERFITTING
    })

    ## COMMAND: python diffusion.py fit --config ./config.yaml
    ## For continue training from a checkpoint: python diffusion.py fit --config ./default_config.yaml --ckpt_path PATH