import math
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            # Upsampling with ConvTranspose3d
            self.conv1 = nn.Conv3d(in_channels=2*in_ch, out_channels=out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            self.transform = nn.ConvTranspose3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))
        else:
            # Downsampling with Conv3d
            self.conv1 = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            self.transform = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv2 = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bnorm1 = nn.BatchNorm3d(out_ch)
        self.bnorm2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[:, :, None, None, None]  # Extend last 3 dimensions for broadcasting
        h = h + time_emb  # Add time embedding to feature maps
        
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Downsample or Upsample
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
    def __init__(self, time_emb_dim, down_channels):
        super().__init__()

        self.down_channels = down_channels # Right part of the UNet
        self.up_channels = self.down_channels[::-1] # Same channels than down_channels, but reversed
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection with Conv3d
        self.conv0 = nn.Conv3d(in_channels=2, out_channels=self.down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i+1], time_emb_dim, up=False) for i in range(len(self.down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i+1], time_emb_dim, up=True) for i in range(len(self.up_channels)-1)])

        self.output = nn.Conv3d(self.up_channels[-1], 2, kernel_size=1)

    def forward(self, X, Q, timestep):
        t = self.time_mlp(timestep)
        
        # Reshape inputs to 5D: (batch_size, channels, frames, joints, spatial_dims) --> Channel indicates if it's position or quaternions.
        batch_size, frames, joints, pos_dims = X.shape
        X = X.view(batch_size, 1, frames, joints, pos_dims)
        batch_size, frames, joints, quaternion_dims = Q.shape
        Q = Q.view(batch_size, 1, frames, joints, quaternion_dims)

        # Padding dimension 4 to be able to concatenate X and Q
        if pos_dims < quaternion_dims:
            # Calculate padding amounts
            padding_size = quaternion_dims - pos_dims
            # Pad X with zeros on the last dimension
            X = F.pad(X, (0, padding_size, 0, 0, 0, 0), mode='constant', value=0)  # Pad the last dimension of X
        else:
            raise AssertionError('pos_dims should be 3 and quaternion_dims 4... Something is wrong')
        
        # Concatenate inputs along channel dimension
        X_and_Q = torch.cat((X, Q), dim=1)
        
        # Initial conv and safety check
        X_and_Q = self.conv0(X_and_Q.float())
        assert X_and_Q.shape[1] == self.down_channels[0], "The first dimension of the data doesn't match with the num of channels of the first step of the network!"
        

        residual_inputs = []
        for down in self.downs:
            X_and_Q = down(X_and_Q, t)
            residual_inputs.append(X_and_Q)
        
        for up in self.ups:
            residual_X_and_Q = residual_inputs.pop()

            if residual_X_and_Q.shape[2:] != X_and_Q.shape[2:]: ### TODO: SUSPICIOUSSSSS
                #print("Entering the suspicious if!!!!!")
                X_and_Q = F.interpolate(X_and_Q, size=residual_X_and_Q.shape[2:])

            X_and_Q = torch.cat((X_and_Q, residual_X_and_Q), dim=1) ## TODO: I don't understand why
            X_and_Q = up(X_and_Q, t)
        
        output = self.output(X_and_Q)
        
        return output