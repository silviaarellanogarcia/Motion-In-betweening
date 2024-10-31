from torch import nn
import math
import torch.nn.functional as F

import torch


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_features)

        # Two fully connected layers in place of Conv1d layers
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bnorm1 = nn.BatchNorm1d(out_features)
        self.bnorm2 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # First Linear Layer
        h = self.bnorm1(self.relu(self.fc1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        #time_emb = time_emb[:, None, :]  # Expand to match x dimensions
        
        # Add time embedding
        h = h + time_emb
        # Second Linear Layer
        h = self.bnorm2(self.relu(self.fc2(h)))
        return h


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


class SimpleMLP(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, time_emb_dim, window, n_joints, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims  # Network dimensions for each layer

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial linear layer to project concatenated X and Q data to the first hidden dimension
        self.fc0 = nn.Linear(window * n_joints * 7, self.hidden_dims[0])

        # MLP layers
        self.layers = nn.ModuleList([MLPBlock(self.hidden_dims[i], self.hidden_dims[i+1], time_emb_dim) 
                                     for i in range(len(self.hidden_dims)-1)])

        # Output layer to project to the original input dimension
        self.output = nn.Linear(self.hidden_dims[-1], window * n_joints * 7)

    def forward(self, X, Q, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        batch_size, frames, joints, pos_dims = X.shape
        X = X.view(batch_size, frames * joints, pos_dims) ## TODO: THIS NEEDS TO BE CHANGED IF I USE MLP
        batch_size, frames, joints, quaternion_dims = Q.shape
        Q = Q.view(batch_size, frames * joints, quaternion_dims) 

        # Concatenate the channels dimensions to be able to pass to the network X and Q at the same time
        X_and_Q = torch.cat((X, Q), dim=2)

        X_and_Q = torch.permute(X_and_Q, (0,2,1))

        # Initial Linear Projection
        X_and_Q = X_and_Q.flatten(start_dim=1)
        h = self.fc0(X_and_Q.float())

        # Pass through the layers
        for layer in self.layers:
            h = layer(h, t)

        # Final Output Layer
        output = self.output(h)

        output = output.view(batch_size, 7, frames*joints)
        
        return output