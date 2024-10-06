import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from inbetweening.data_processing.extract import bvh_to_item
from inbetweening.data_processing.process_data import Lafan1Dataset
from inbetweening.model.unet import SimpleUnet

def get_scheduler(schedule_name, n_diffusion_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Define a noise scheduler
    """
    if schedule_name == 'linear':
        return torch.linspace(beta_start, beta_end, n_diffusion_timesteps, dtype=torch.float64)
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
    def __init__(self, betas, lr, device='cpu'):
        super().__init__() # Initialize the parent's class before initializing any child
        self.betas = betas
        self.n_diffusion_timesteps = self.betas.shape[0]
        
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0) # Cumprod stands for cumulative product. It tells us how much info remains.
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) ## Adds a 1 at the beginning. At t=0 we have all the info
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.model = SimpleUnet()
        self.lr = lr
        self.batch_size = 1

        self.bvh_path = "/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data2" ## TODO: Generalize later
        self.window = 50
        self.offset = 20
    
    def train_dataloader(self):
        dataset = Lafan1Dataset(self.bvh_path, window=self.window, offset=self.offset, train=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        dataset = Lafan1Dataset(self.bvh_path, window=self.window, offset=self.offset, train=False) ## TODO: Create a validation dataset
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    def test_dataloader(self):
        dataset = dataset = Lafan1Dataset(self.bvh_path, window=self.window, offset=self.offset, train=False)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False
        )
        return loader

    def forward_diffusion_sample(self, X_0, Q_0, t):
        """ 
        Takes a motion sequence and a timestep as input and returns the noisy version of it
        """ 
        ## Apply noise to position --> It only needs to be applied on the root, not the offsets.
        noise_X = torch.randn_like(X_0)
        sqrt_alphas_cumprod_t_X = get_index_from_list(self.sqrt_alphas_cumprod, t, X_0.shape)  ## Alpha with an overline in the notation
        sqrt_one_minus_alphas_cumprod_t_X = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, X_0.shape)

        noisy_X_0 = sqrt_alphas_cumprod_t_X.to(self.device) * X_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t_X.to(self.device) * noise_X.to(self.device)

        ## Apply noise to quaternions
        noise_Q = torch.randn_like(Q_0)
        sqrt_alphas_cumprod_t_Q = get_index_from_list(self.sqrt_alphas_cumprod, t, Q_0.shape) 
        sqrt_one_minus_alphas_cumprod_t_Q = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, Q_0.shape)
        noisy_Q_0 = sqrt_alphas_cumprod_t_Q.to(self.device) * Q_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t_Q.to(self.device) * noise_Q.to(self.device)

        # Normalize the quaternions to ensure they are valid unit quaternions
        noisy_Q_0 = F.normalize(noisy_Q_0, dim=-1)
        
        return noisy_X_0, noisy_Q_0, noise_X, noise_Q
    
    
    def get_loss(self, model, X_0, Q_0, t):
        """
        Compute the loss between the predicted noise and the actual noise for both positions (X_0) and quaternions (Q_0).
        """
        ## TODO: FIX TO INCLUDE X_0 TOO!
        noisy_X_0, noisy_Q_0, noise_X, noise_Q = self.forward_diffusion_sample(X_0, Q_0, t)
        
        # Predict noise for both positional and quaternion data
        noise_pred = model(noisy_X_0, noisy_Q_0, t)

        # Reshape the noise so that it has the same structure as the noise prediction.
        batch_size, frames, joints, quaternion_dims = noise_X.shape
        noise_X = noise_X.view(batch_size, quaternion_dims, frames * joints)
        batch_size, frames, joints, quaternion_dims = noise_Q.shape
        noise_Q = noise_Q.view(batch_size, quaternion_dims, frames * joints)

        # Concatenate the channels dimensions to consider X and Q at the same time
        noise_X_and_Q = torch.cat((noise_X, noise_Q), dim=1)
        
        # Calculate the loss (you can change L1 to MSE if needed)
        loss_X_and_Q = F.l1_loss(noise_X_and_Q, noise_pred)

        total_loss = loss_X_and_Q
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        # Batch processing
        X_0 = batch['X']
        Q_0 = batch['Q']
        t = torch.randint(0, self.n_diffusion_timesteps, (Q_0.shape[0],), device=self.device).long()

        # Calculate loss
        loss = self.get_loss(self.model, X_0, Q_0, t)
        
        # Log loss
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    beta_start = 0.0001
    beta_end = 0.02
    n_diffusion_timesteps = 300

    # Get beta scheduler
    betas = get_scheduler('linear', n_diffusion_timesteps, beta_start, beta_end)

    model = DiffusionModel(betas, lr=0.001, device='cpu')
    trainer = pl.Trainer(max_epochs=150, precision="bf16-mixed") ### TODO: ASK IF BF16-MIXER IS OKAY.
    trainer.fit(model)