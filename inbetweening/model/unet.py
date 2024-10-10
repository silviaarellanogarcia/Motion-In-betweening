from torch import nn
import math
import torch.nn.functional as F

import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 3, 1, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[:, :, None]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2 # One half of the dims uses sin and the other cos
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        frames_per_joints = 50 * 22 ## TODO: window_size*njoints. Generalize

        self.down_channels = (1024, 512, 256, 128, 64) # TODO: Right part of the UNet ## Check that 1024 is not higher than original data dim !!!!! --> [1, 7, 1100] to [1,1024,1100]
        self.up_channels = (64, 128, 256, 512, 1024) # Left part of the UNet
        time_emb_dim = 32 ## TODO: CHECK! Try different values. It's an hyperparameter!

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv1d(in_channels=frames_per_joints, out_channels=self.down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i+1], time_emb_dim, up=False) for i in range(len(self.down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i+1], time_emb_dim, up=True) for i in range(len(self.up_channels)-1)])
        
        self.output = nn.Conv1d(self.up_channels[-1], frames_per_joints, 1)

    def forward(self, X, Q, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        batch_size, frames, joints, quaternion_dims = X.shape
        X = X.view(batch_size, quaternion_dims, frames * joints)
        batch_size, frames, joints, quaternion_dims = Q.shape
        Q = Q.view(batch_size, quaternion_dims, frames * joints) 

        # Concatenate the channels dimensions to be able to pass to the network X and Q at the same time
        X_and_Q = torch.cat((X, Q), dim=1)

        # Change the order of the dimensions to increase/decrease the frames*joints dimensions.
        X_and_Q = X_and_Q.permute(0, 2, 1)

        # Initial conv and safety check
        X_and_Q = self.conv0(X_and_Q.float())
        assert X_and_Q.shape[1] == self.down_channels[0], "The first dimension of the data doesn't match with the num of channels of the first step of the network!"

        # Unet
        residual_inputs = []
        for down in self.downs:
            X_and_Q = down(X_and_Q, t)
            residual_inputs.append(X_and_Q)
        for up in self.ups:
            residual_X_and_Q = residual_inputs.pop()

            if residual_X_and_Q.shape[2] > X_and_Q.shape[2]:
                # Pad Q to match the shape of residual_Q and allow concatenation
                pad_amount = (0, residual_X_and_Q.shape[2] - X_and_Q.shape[2]) # No padding on the left side, just the right.
                X_and_Q = F.pad(X_and_Q, pad_amount)

            # Add residual x as additional channels
            X_and_Q = torch.cat((X_and_Q, residual_X_and_Q), dim=1)           
            X_and_Q = up(X_and_Q, t)

        output = self.output(X_and_Q)
        
        return output

if __name__ == '__main__':
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model