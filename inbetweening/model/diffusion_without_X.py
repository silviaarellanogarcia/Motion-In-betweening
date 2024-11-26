import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml

from inbetweening.data_processing.process_data import Lafan1DataModule, Lafan1Dataset
from inbetweening.model.mlp import SimpleMLP
from inbetweening.model.unet import SimpleUnetJustAngles
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

# from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines

def get_scheduler(schedule_name, n_diffusion_timesteps, beta_start, beta_end):
    """
    Define a noise scheduler
    """
    if schedule_name == 'linear':
        scale = 1000 / n_diffusion_timesteps
        new_beta_start = scale * 0.0001 ### TODO: Check if this works and if it does, adjust the parameters passed
        new_beta_end = scale * 0.02
        return torch.linspace(new_beta_start, new_beta_end, n_diffusion_timesteps, dtype=torch.float64)
    else:
        raise NotImplementedError(f"The scheduler: {schedule_name} is not implemented. Try to use linear")
    

def get_index_from_list(vals, t, x_shape):
    """
    Retrieves a value of precomputed noise parameters at a specific timestep t.
    This is useful when we sample a random timestep t and want to pass a noisified version of the image 
    (based on the corresponding noise parameters at time t) to the model.
    """
    
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()) # Retrieves values from the last dimension of vals. ### TODO: CHECK THIS
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionModel(pl.LightningModule):
    def __init__(self, beta_start: float, beta_end: float, n_diffusion_timesteps: int, lr: float, gap_size: int, type_masking: str, time_emb_dim: int, window: int, n_joints: int, down_channels: list[int], type_model: str, kernel_size: int, step_threshold: int, max_gap_size: int):
        super().__init__() # Initialize the parent's class before initializing any child

        # Get beta scheduler
        betas = get_scheduler('linear', n_diffusion_timesteps, beta_start, beta_end)
        self.betas = betas
        self.n_diffusion_timesteps = n_diffusion_timesteps
        
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0) # Cumprod stands for cumulative product. It tells us how much info remains.
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) ## Adds a 1 at the beginning. At t=0 we have all the info
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.type_model = type_model
        
        self.kernel_size = kernel_size

        self.model = SimpleUnetJustAngles(time_emb_dim, window, n_joints, down_channels, kernel_size)
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


    def forward_diffusion_sample(self, Q_0, t, masked_frames):
        """ 
        Takes a motion sequence and a timestep as input and returns the noisy version of it
        Important! It should only apply noise to the masked frames.
        """ 
        ## Apply noise to angles
        noise_Q = torch.randn_like(Q_0)
        sqrt_alphas_cumprod_t_Q = get_index_from_list(self.sqrt_alphas_cumprod, t, Q_0.shape) 
        sqrt_one_minus_alphas_cumprod_t_Q = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, Q_0.shape)

        noisy_Q_0 = Q_0.clone()
        noisy_Q_0[:, masked_frames, :, :] = (sqrt_alphas_cumprod_t_Q.to(self.device) * Q_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t_Q.to(self.device) * noise_Q.to(self.device))[:, masked_frames, :, :].float()
        
        return noisy_Q_0, noise_Q
    
    def sample_timestep(self, noisy_Q, t):
        """
        Calls the model to predict the noise in the motion sequence and returns the denoised image. 
        Applies noise to this motion sequence, if we are not in the last step yet.
        """
        betas_t = get_index_from_list(self.betas, t, noisy_Q.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, noisy_Q.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, noisy_Q.shape)

        noise_Q_pred = self.model(noisy_Q, t) ### I will have X and Q together and I have to separate them.
        noise_Q_pred = torch.permute(noise_Q_pred, (0,2,1))

        # Convert back to the shape (1, 50, 22, angle_dim) --> TODO: BE CAREFUL!
        batch_size = t.shape[0]
        noise_Q_pred = noise_Q_pred.view(batch_size, self.window, self.n_joints, 6)

        # Call model (current image - noise prediction)
        model_mean_Q = sqrt_recip_alphas_t * (noisy_Q - betas_t * noise_Q_pred / sqrt_one_minus_alphas_cumprod_t) ## This corresponds to eq. 11

        posterior_variance_t = get_index_from_list(self.posterior_variance, t, noisy_Q.shape)
        
        if t == 0:
            ## Current model_mean_X and model_mean_Q contain all the frames. In the training step I should only keep the ones that correspond to the gap and 
            ## incorporate these into the complete sequence.
            return model_mean_Q
        else:
            ## These X and Q minus one contain everything, but I should only keep the part that corresponds to the gap, and concatenate that to the original motion.
            Q_minus_one = model_mean_Q + torch.sqrt(posterior_variance_t) * torch.randn_like(model_mean_Q) 
            return Q_minus_one
    
    
    def get_loss(self, model, Q_0, t, masked_frames):
        """
        Compute the loss between the predicted noise and the actual noise for both positions (X_0) and quaternions (Q_0).
        """
        noisy_Q_0, noise_Q = self.forward_diffusion_sample(Q_0, t, masked_frames)
        
        # Predict noise for both positional and quaternion data

        noise_pred = model(noisy_Q_0, t)

        # Reshape the noise so that it has the same structure as the noise prediction.
        batch_size, frames, joints, angle_dims = noise_Q.shape
        noise_Q = noise_Q.view(batch_size, frames, joints * angle_dims)
        noise_Q = torch.permute(noise_Q, (0,2,1))
        
        # Create a tensor for the masked frames
        masked_frames_tensor = torch.tensor(masked_frames).view(-1, 1)
        masked_frames_tensor = masked_frames_tensor.view(-1)

        # Calculate the loss
        loss_Q = F.mse_loss(noise_Q[:, :, masked_frames_tensor], noise_pred[:, :, masked_frames_tensor], reduction='sum')
        
        return loss_Q
    
    def training_step(self, batch, batch_idx):
        if self.steps_since_last_gap_increase >= self.step_threshold:  # TODO: Change 5 with a parameter indicating the maximum gap
            self.gap_size = min(self.gap_size + 1, self.max_gap_size)  # TODO: Adjust maximum gap parameter
            self.step_threshold += 5000
            self.steps_since_last_gap_increase = 0

        Q_0 = batch['Q']

        t = torch.randint(0, self.n_diffusion_timesteps, (Q_0.shape[0],), device=self.device).long()
        masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

        # Calculate loss
        loss_Q = self.get_loss(self.model, Q_0, t, masked_frames)
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
        Q_0 = batch['Q']
        t = torch.randint(0, self.n_diffusion_timesteps, (Q_0.shape[0],), device=self.device).long()

        # Masking
        masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

        # Q_input = Q_0 - Q_0[:, masked_frames[0] - 1, :, :].unsqueeze(1) ## TODO: TRY THIS THING LATER, NOW JUST PREDICT THE Q

        # Calculate loss
        loss_Q = self.get_loss(self.model, Q_0, t, masked_frames)
        total_loss = (loss_Q) / Q_0.shape[0]
        
        # Log loss
        self.log('validation_total_loss', total_loss, prog_bar=True, on_step=True) # We divide the loss by the batch size

        return total_loss
    
    def generate_samples(self, Q_0):
        self.eval()
        with torch.no_grad():
            Q_0 = Q_0.unsqueeze(0) ## This adds the batch dimension

            t = torch.full((Q_0.shape[0],), self.n_diffusion_timesteps - 1, device=self.device).long()

            # Masking
            masked_frames = self.masking(n_frames=Q_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

            # Calculate loss
            noisy_Q_0, _ = self.forward_diffusion_sample(Q_0, t, masked_frames)

            for step in reversed(range(self.n_diffusion_timesteps)):
                t_step = torch.tensor([step], device=self.device).long()

                # Denoise positions and angles
                denoised_Q_complete_seq = self.sample_timestep(noisy_Q_0, t_step)
                noisy_Q_0[:, masked_frames, :, :] = denoised_Q_complete_seq[:, masked_frames, :, :].float()

        return noisy_Q_0[0], masked_frames


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
            'name': 'my_model_only_Q',
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
    LightningCLI(DiffusionModel, 
                 Lafan1DataModule, 
                 trainer_defaults={
                     'logger': logger_config,
                     'callbacks': [checkpoint_callback],
                    #  'overfit_batches': 1 ## TODO: AT SOME POINT REMOVE THE OVERFITTING
    })

    ## COMMAND: python diffusion.py fit --config ./config.yaml
    ## For continue training from a checkpoint: python diffusion.py fit --config ./default_config.yaml --ckpt_path PATH