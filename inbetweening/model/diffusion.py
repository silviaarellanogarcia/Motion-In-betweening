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
    def __init__(self, beta_start: float, beta_end: float, n_diffusion_timesteps: int, lr: float, gap_size: int, type_masking: str, time_emb_dim: int, window: int, n_joints: int, down_channels: list[int], type_model: str, kernel_size: int):
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

        if type_model == 'unet':
            self.model = SimpleUnet(time_emb_dim, window, n_joints, down_channels, kernel_size)
        else:
            ## In this case down_channels refers to the hidden dimensions
            self.model = SimpleMLP(time_emb_dim, window, n_joints, down_channels)
        self.lr = lr
        self.window = window
        self.n_joints = n_joints

        self.gap_size = gap_size
        self.type_masking = type_masking

        ## USEFUL FOR FINDING IF IT WORKS CORRECTLY, BUT ERASE LATER (TODO: DELETE)
        # self.FIXED_NOISE_X = torch.randn((256, window, n_joints, 3))
        # self.FIXED_NOISE_Q = torch.randn((256, window, n_joints, 4))
    
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

        return masked_frames


    def forward_diffusion_sample(self, X_0, Q_0, t, masked_frames):
        """ 
        Takes a motion sequence and a timestep as input and returns the noisy version of it
        Important! It should only apply noise to the masked frames.
        """ 
        ## Apply noise to position --> It only needs to be applied on the root, not the offsets.
        noise_X = torch.randn_like(X_0)
        # noise_X = self.FIXED_NOISE_X.cuda() ## TODO: Delete this when I finish debugging
        sqrt_alphas_cumprod_t_X = get_index_from_list(self.sqrt_alphas_cumprod, t, X_0.shape)  ## Alpha with an overline in the notation
        sqrt_one_minus_alphas_cumprod_t_X = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, X_0.shape)

        noisy_X_0 = X_0.clone()
        noisy_X_0[:, masked_frames, 0, :] = (sqrt_alphas_cumprod_t_X.to(self.device) * X_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t_X.to(self.device) * noise_X.to(self.device))[:, masked_frames, 0, :].float()

        ## Apply noise to quaternions
        noise_Q = torch.randn_like(Q_0)
        # noise_Q = self.FIXED_NOISE_Q.cuda() ## TODO: Delete this when I finish debugging
        sqrt_alphas_cumprod_t_Q = get_index_from_list(self.sqrt_alphas_cumprod, t, Q_0.shape) 
        sqrt_one_minus_alphas_cumprod_t_Q = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, Q_0.shape)

        noisy_Q_0 = Q_0.clone()
        noisy_Q_0[:, masked_frames, :, :] = (sqrt_alphas_cumprod_t_Q.to(self.device) * Q_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t_Q.to(self.device) * noise_Q.to(self.device))[:, masked_frames, :, :].float()
        
        return noisy_X_0, noisy_Q_0, noise_X, noise_Q
    
    def sample_timestep(self, noisy_X, noisy_Q, t):
        """
        Calls the model to predict the noise in the motion sequence and returns the denoised image. 
        Applies noise to this motion sequence, if we are not in the last step yet.
        """
        betas_t = get_index_from_list(self.betas, t, noisy_X.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, noisy_X.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, noisy_X.shape)

        noise_pred = self.model(noisy_X, noisy_Q, t) ### I will have X and Q together and I have to separate them.
        noise_X_pred = noise_pred[:, :(self.n_joints * 3), :]
        noise_Q_pred = noise_pred[:, (self.n_joints * 3):, :]

        noise_X_pred = torch.permute(noise_X_pred, (0,2,1))
        noise_Q_pred = torch.permute(noise_Q_pred, (0,2,1))

        # Convert back to the shape (1, 50, 22, 3) --> TODO: BE CAREFUL!
        batch_size = t.shape[0]
        noise_X_pred = noise_X_pred.view(batch_size, self.window, self.n_joints, 3)
        noise_Q_pred = noise_Q_pred.view(batch_size, self.window, self.n_joints, 4)

        # Call model (current image - noise prediction)
        model_mean_X = sqrt_recip_alphas_t * (noisy_X - betas_t * noise_X_pred / sqrt_one_minus_alphas_cumprod_t) ## This corresponds to eq. 11
        model_mean_Q = sqrt_recip_alphas_t * (noisy_Q - betas_t * noise_Q_pred / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = get_index_from_list(self.posterior_variance, t, noisy_X.shape)
        
        if t == 0:
            ## Current model_mean_X and model_mean_Q contain all the frames. In the training step I should only keep the ones that correspond to the gap and 
            ## incorporate these into the complete sequence.
            return model_mean_X, model_mean_Q
        else:
            ## These X and Q minus one contain everything, but I should only keep the part that corresponds to the gap, and concatenate that to the original motion.
            X_minus_one = model_mean_X + torch.sqrt(posterior_variance_t) * torch.randn_like(model_mean_X) ## Equation 4 in algorihm 2
            Q_minus_one = model_mean_Q + torch.sqrt(posterior_variance_t) * torch.randn_like(model_mean_Q) 
            ### TODO: Maybe it's better to predict the clean motion instead of the noise (predict x_{0} directly, not x_{t-1})

            return X_minus_one, Q_minus_one
    
    
    def get_loss(self, model, X_0, Q_0, t, masked_frames):
        """
        Compute the loss between the predicted noise and the actual noise for both positions (X_0) and quaternions (Q_0).
        """
        noisy_X_0, noisy_Q_0, noise_X, noise_Q = self.forward_diffusion_sample(X_0, Q_0, t, masked_frames)
        
        # Predict noise for both positional and quaternion data
        noise_pred = model(noisy_X_0, noisy_Q_0, t)

        # Reshape the noise so that it has the same structure as the noise prediction.
        batch_size, frames, joints, position_dims = noise_X.shape
        noise_X = noise_X.view(batch_size, frames, joints * position_dims)

        batch_size, frames, joints, quaternion_dims = noise_Q.shape
        noise_Q = noise_Q.view(batch_size, frames, joints * quaternion_dims)

        # Concatenate the channels dimensions to consider X and Q at the same time
        noise_X_and_Q = torch.cat((noise_X, noise_Q), dim=2)
        noise_X_and_Q = torch.permute(noise_X_and_Q, (0,2,1))
        
        # Create a tensor for the masked frames
        masked_frames_tensor = torch.tensor(masked_frames).view(-1, 1)
        masked_frames_tensor = masked_frames_tensor.view(-1)

        # Calculate the loss
        loss_X = F.mse_loss(noise_X_and_Q[:, :(joints * 3), masked_frames_tensor], noise_pred[:, :(joints * 3), masked_frames_tensor], reduction='sum')
        loss_Q = F.mse_loss(noise_X_and_Q[:, (joints * 3):, masked_frames_tensor], noise_pred[:, (joints * 3):, masked_frames_tensor], reduction='sum')
        
        return loss_X, loss_Q
    
    def training_step(self, batch, batch_idx):
        # Batch processing
        X_0 = batch['X']
        Q_0 = batch['Q']
        t = torch.randint(0, self.n_diffusion_timesteps, (Q_0.shape[0],), device=self.device).long()

        # Masking
        masked_frames = self.masking(n_frames=X_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

        # Calculate loss
        loss_X, loss_Q = self.get_loss(self.model, X_0, Q_0, t, masked_frames)
        total_loss = ((1/X_0.shape[2] * loss_X) + loss_Q) / X_0.shape[0]
        
        # Log loss
        self.log('train_loss_X', loss_X/X_0.shape[0], prog_bar=True, on_step=True) # We divide the loss by the batch size
        self.log('train_loss_Q', loss_Q/X_0.shape[0], prog_bar=True, on_step=True) # We divide the loss by the batch size
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Batch processing
        X_0 = batch['X']
        Q_0 = batch['Q']
        t = torch.randint(0, self.n_diffusion_timesteps, (Q_0.shape[0],), device=self.device).long()

        # Masking
        masked_frames = self.masking(n_frames=X_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

        # Calculate loss
        loss_X, loss_Q = self.get_loss(self.model, X_0, Q_0, t, masked_frames)
        total_loss = ((1/X_0.shape[2] * loss_X) + loss_Q) / X_0.shape[0]
        
        # Log loss
        self.log('validation_loss_X', loss_X / X_0.shape[0], prog_bar=True, on_step=True) # We divide the loss by the batch size
        self.log('validation_loss_Q', loss_Q / X_0.shape[0], prog_bar=True, on_step=True) # We divide the loss by the batch size
        self.log('validation_total_loss', total_loss, prog_bar=True, on_step=True)

        return total_loss
    
    def generate_samples(self, X_0, Q_0):
        self.eval()
        with torch.no_grad():
            X_0 = X_0.unsqueeze(0) ## This adds the batch dimension
            Q_0 = Q_0.unsqueeze(0) ## This adds the batch dimension

            t = torch.full((X_0.shape[0],), self.n_diffusion_timesteps - 1, device=self.device).long()

            # Masking
            masked_frames = self.masking(n_frames=X_0.shape[1], gap_size=self.gap_size, type=self.type_masking)

            # Calculate loss
            noisy_X_0, noisy_Q_0, _, _ = self.forward_diffusion_sample(X_0, Q_0, t, masked_frames)

            for step in reversed(range(self.n_diffusion_timesteps)):
                t_step = torch.tensor([step], device=self.device).long()

                # Denoise positions and quaternions
                denoised_X_complete_seq, denoised_Q_complete_seq = self.sample_timestep(noisy_X_0, noisy_Q_0, t_step)
                noisy_X_0[:, masked_frames, :, :] = denoised_X_complete_seq[:, masked_frames, :, :].float()
                noisy_Q_0[:, masked_frames, :, :] = denoised_Q_complete_seq[:, masked_frames, :, :].float()

                # Normalize quaternions to ensure they remain valid unit quaternions
                noisy_Q_0 = F.normalize(noisy_Q_0, dim=-1)

            # Normalize the quaternions to ensure they are valid unit quaternions
            noisy_Q_0 = F.normalize(noisy_Q_0, dim=-1) ## TODO: Check that the samples that weren't modified remain the same

        return noisy_X_0[0], noisy_Q_0[0]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    print("AM I USING GPU? ", torch.cuda.is_available())
    logger_config = {
        'class_path': 'pytorch_lightning.loggers.TensorBoardLogger',
        'init_args': {                      # Use init_args instead of params
            'save_dir': 'lightning_logs',
            'name': 'my_model_scaling',
            'version': None
        }
    }

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,  # Keep 10 checkpoints
        monitor='validation_total_loss',
        mode="min"
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