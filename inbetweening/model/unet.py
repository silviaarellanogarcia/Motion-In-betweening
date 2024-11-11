from torch import nn
import math
import torch.nn.functional as F

import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)  

        if up:
            self.conv1 = nn.Conv1d(2 * in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            # Use ConvTranspose1d for upsampling (increasing sequence length)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2)  # Downsampling with stride instead of MaxPool
            self.transform = nn.Identity()
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.mish = nn.Mish() ## An activation function similar to ReLU
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.mish(self.conv1(x)))
        # Time embedding
        time_emb = self.mish(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[:, :, None]
        # Add time embedding
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.mish(self.conv2(h)))
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
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, time_emb_dim, window, n_joints, down_channels, kernel_size):
        super().__init__()
        self.n_joints = n_joints
        self.down_channels = down_channels # Right part of the UNet
        self.up_channels = self.down_channels[::-1] # Same channels than down_channels, but reversed
        print(self.up_channels)
        self.time_emb_dim = time_emb_dim
        self.kernel_size = kernel_size
        self.n_dimensions = 6

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv1d(in_channels=n_joints*self.n_dimensions, out_channels=self.down_channels[0], kernel_size=kernel_size, padding=kernel_size // 2)

        # Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i+1], self.kernel_size, time_emb_dim, up=False) for i in range(len(self.down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i+1], self.kernel_size, time_emb_dim, up=True) for i in range(len(self.up_channels)-1)])
        
        self.output = nn.Conv1d(self.up_channels[-1], n_joints*self.n_dimensions, 1)
        self.output_linear = nn.Linear(n_joints*self.n_dimensions, n_joints*self.n_dimensions)

    def forward(self, Q, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        batch_size, frames, joints, quaternion_dims = Q.shape
        Q = Q.view(batch_size, frames, joints * quaternion_dims) 
        Q = torch.permute(Q, (0,2,1))

        # Initial conv and safety check
        Q = self.conv0(Q.float())
        assert Q.shape[1] == self.down_channels[0], "The first dimension of the data doesn't match with the num of channels of the first step of the network!"

        # Unet
        residual_inputs = []
        for down in self.downs:
            Q = down(Q, t)
            residual_inputs.append(Q)
        for up in self.ups:
            residual_Q = residual_inputs.pop()

            # Ensure both tensors have the same length
            min_len = min(Q.shape[2], residual_Q.shape[2])
            Q = Q[:, :, :min_len]
            residual_Q = residual_Q[:, :, :min_len]

            # Concatenate along the channel dimension
            Q = torch.cat((Q, residual_Q), dim=1)
            Q = up(Q, t)

        output = self.output(Q)

        output = output.permute(0, 2, 1)
        output = self.output_linear(output)
        output = output.permute(0, 2, 1)
        
        return output

if __name__ == '__main__':
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model