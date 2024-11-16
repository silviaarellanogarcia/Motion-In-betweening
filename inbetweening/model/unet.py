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
        half_dim = self.dim // 2  # One half of the dims uses sin and the other cos
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    Predicts Ortho6D angles (6D representation of rotation) for each joint, using positions (X) as auxiliary information.
    """
    def __init__(self, time_emb_dim, window, n_joints, down_channels, kernel_size):
        super().__init__()
        self.n_joints = n_joints
        self.down_channels = down_channels  # Right part of the UNet
        self.up_channels = self.down_channels[::-1]  # Same channels as down_channels but reversed
        print(self.up_channels)
        self.time_emb_dim = time_emb_dim
        self.kernel_size = kernel_size
        self.input_n_dimensions = 6 + 3  # Ortho6D: 6 components per joint for rotation (3 for axis, 3 for magnitude)
        self.output_n_dimensions = 6 # I only want to output the angles

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection (taking angles into account, no positions in the output)
        self.conv0 = nn.Conv1d(in_channels=n_joints * self.input_n_dimensions, out_channels=self.down_channels[0], kernel_size=kernel_size, padding=kernel_size // 2)

        # Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i + 1], self.kernel_size, time_emb_dim, up=False) for i in range(len(self.down_channels) - 1)])

        # Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i + 1], self.kernel_size, time_emb_dim, up=True) for i in range(len(self.up_channels) - 1)])

        # Final output layer: predicting Ortho6D (6D) angles for each joint
        self.output = nn.Conv1d(self.up_channels[-1], n_joints * self.output_n_dimensions, 1)  # Ortho6D angles per joint
        self.output_linear = nn.Linear(n_joints * self.output_n_dimensions, n_joints * self.output_n_dimensions)  # Linear layer to output final angles

    def forward(self, X, Q, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        batch_size, frames, joints, pos_dims = X.shape
        X = X.view(batch_size, frames, joints * pos_dims)  # Flatten positions (X)

        batch_size, frames, joints, angle_dims = Q.shape
        Q = Q.view(batch_size, frames, joints * angle_dims)  # Flatten Ortho6D angles (Q)

        # Concatenate positions (X) and Ortho6D angles (Q) along the joint dimension
        X_and_Q = torch.cat((X, Q), dim=2)  # X and Q are concatenated along the channels axis
        X_and_Q = torch.permute(X_and_Q, (0, 2, 1))  # Change to (batch_size, channels, frames)

        # Initial conv and safety check
        X_and_Q = self.conv0(X_and_Q.float())
        assert X_and_Q.shape[1] == self.down_channels[0], "Mismatch in channel size!"

        # Unet: Downsampling and Upsampling
        residual_inputs = []
        for down in self.downs:
            X_and_Q = down(X_and_Q, t)
            residual_inputs.append(X_and_Q)

        for up in self.ups:
            residual_X_and_Q = residual_inputs.pop()

            # Ensure both tensors have the same length
            min_len = min(X_and_Q.shape[2], residual_X_and_Q.shape[2])
            X_and_Q = X_and_Q[:, :, :min_len]
            residual_X_and_Q = residual_X_and_Q[:, :, :min_len]

            # Concatenate along the channel dimension
            X_and_Q = torch.cat((X_and_Q, residual_X_and_Q), dim=1)
            X_and_Q = up(X_and_Q, t)

        # Final output: We only care about the Ortho6D angles (Q), not positions (X)
        output = self.output(X_and_Q)
        output = output.permute(0, 2, 1)  # Revert to (batch_size, frames, joints * 6)
        output = self.output_linear(output)
        output = output.permute(0, 2, 1)  # Final output: (batch_size, frames, joints * 6)

        return output


if __name__ == '__main__':
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model