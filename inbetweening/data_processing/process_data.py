import pickle
import os
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from inbetweening.data_processing.extract import get_lafan1_set
import pymotion.rotations.ortho6d as sixd

class Lafan1Dataset(Dataset):
    """LAFAN1 Dataset class."""

    def __init__(self, data_dir: str, window: int, offset: int, scaling: int, train: bool=True, val: bool=False, test: bool=False):
        """
        Args:
            data_dir (string): Directory with the dataset.
            window (int): Size of the sliding window
            offset (int): Offset between windows (in timesteps)
            train (boolean): Indicates if the dataset is for training or testing

        Outputs:
            X: global positions
            Q: global quaternions
            parents: list of parent indices defining the bone hierarchy
            contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
            contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
        """

        self.data_dir = data_dir
        self.window = window
        self.offset = offset

        self.scaling = scaling

        self.training_mean_X = None
        self.training_std_X = None

        self.actors_train = ['subject1', 'subject2', 'subject3']
        self.act_num_train = '123'

        # train info:
        if train:
            self.actors = self.actors_train
            act_num = self.act_num_train
        elif val:
            self.actors = ['subject4'] ## TODO: This could be modified. Check if the amount of data of subject 4 is enough.
            act_num = '4'
        elif test:
            self.actors = ['subject5']
            act_num = '5'
        

        # Load the corresponding data
        filename_data = './pickle_data/lafan1_data_actors_' + act_num + '_win_' + str(self.window) + '_off_' + str(self.offset) + '.pkl'
        self.load_or_create_data_pickle(filename_data)

        # Loadd mean and std
        filename_stats = './pickle_data/stats.pkl'

        if self.training_mean_X is None or self.training_std_X is None:
            if os.path.isfile(filename_stats):
                with open(filename_stats, 'rb') as f:
                    self.training_mean_X, self.training_std_X = pickle.load(f)
            else: 
                filename_train_data = './pickle_data/lafan1_data_actors_' + self.act_num_train + '_win_' + str(self.window) + '_off_' + str(self.offset) + '.pkl'
                if not os.path.isfile(filename_train_data):
                    raise FileNotFoundError(f"The training data file {filename_train_data} does not exist!")
                with open(filename_train_data, 'rb') as f:
                    train_X, _, _, _, _, _ = pickle.load(f) 
                # Load training dataset and compute mean and var
                self.training_mean_X = np.mean(train_X, axis=(0, 1))
                self.training_std_X = np.std(train_X, axis=(0, 1))

                # Save the stats
                with open(filename_stats, 'wb') as f:
                    pickle.dump((self.training_mean_X, self.training_std_X), f)


    def __len__(self):
        return len(self.X) ## Returns the number of sequences.


    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the sample to retrieve
        """
        X = self.X[idx]
        Q = self.Q[idx]
        parents = self.parents
        contacts_l = self.contacts_l[idx]
        contacts_r = self.contacts_r[idx]

        assert self.training_mean_X is not None, "Mean X is not set!"
        assert self.training_std_X is not None, "Standard deviation of X is not set!"

        # Normalize the global position (-1, 1)
        X = (X - self.training_mean_X) / (self.training_std_X + 1e-8)
        X = X * self.scaling # TODO: Check this and tune scaling value

        # Convert quaternions to Ortho6D
        Q_rotations_tensor = torch.from_numpy(Q)
        Q = sixd.from_quat(Q_rotations_tensor) # Shape (frames, joints, 3, 2)
        Q = Q.reshape(Q.shape[0], Q.shape[1], -1) # Shape (frames, joints, 6)

        sample = {
            'X': torch.tensor(X, dtype=torch.float32),
            'Q': torch.tensor(Q, dtype=torch.float32),
            'parents': torch.tensor(parents, dtype=torch.int64),
            'contacts_l': torch.tensor(contacts_l, dtype=torch.bool),
            'contacts_r': torch.tensor(contacts_r, dtype=torch.bool)
        }

        return sample
    
    def load_or_create_data_pickle(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                self.X, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map = pickle.load(f)
                print('Dataset loaded! (', filename, ')')
        else:
            # Load the dataset using the existing get_lafan1_set function --> ATTENTION! WE KEEP THE GLOBAL Q. FOR THE X WE DON'T CARE, SINCE THE ROOT IS ALWAYS GLOBAL
            self.X, _, _, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map = get_lafan1_set(self.data_dir, self.actors, self.window, self.offset)

            # with open(filename, 'wb') as f:
            with open(filename, 'wb') as f:
                pickle.dump((self.X, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map), f)
                print('Dataset saved as ', filename)
    
    
class Lafan1DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, window: int, offset: int, scaling: int):
        """
        Args:
            data_dir (string): Directory with the dataset.
            window (int): Size of the sliding window
            offset (int): Offset between windows (in timesteps)
        """
        super().__init__()  # Call the parent class's initializer
        self.data_dir = data_dir
        self.window = window
        self.offset = offset
        self.batch_size = batch_size
        self.scaling = scaling

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False
        )
        return loader


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Lafan1Dataset(self.data_dir, window=self.window, offset=self.offset, scaling=self.scaling, train=True, val=False, test=False)
            self.val_dataset = Lafan1Dataset(self.data_dir, window=self.window, offset=self.offset, scaling=self.scaling, train=False, val=True, test=False)

        elif stage == 'test':
            self.test_dataset = Lafan1Dataset(self.data_dir, window=self.window, offset=self.offset, scaling=self.scaling, train=False, val=False, test=True)
    
if __name__ == "__main__":
    bvh_path = "/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data1"  # Update this with the actual path
    actors = ['Actor1', 'Actor2']  # Replace with actual actor names in your dataset

    # Initialize the dataset
    dataset = Lafan1Dataset(bvh_path, window=50, offset=20, train=True)

    # Test by retrieving a sample from the dataset
    sample_idx = 0  # Test with the first sample

    sample = dataset[sample_idx]

    # Print out details to check if the data is loaded correctly
    print("Positions shape:", sample['X'].shape)
    print("Rotations shape:", sample['Q'].shape)
    print("Parents:", sample['parents'])
    print("Left Foot Contacts shape:", sample['contacts_l'].shape)
    print("Right Foot Contacts shape:", sample['contacts_r'].shape)
