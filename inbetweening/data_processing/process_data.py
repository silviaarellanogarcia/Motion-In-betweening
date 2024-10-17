import pickle
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from inbetweening.data_processing.extract import get_lafan1_set

class Lafan1Dataset(Dataset):
    """LAFAN1 Dataset class."""

    def __init__(self, data_dir: str, window: int, offset: int, train: bool=True, val: bool=False):
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

        if train:
            self.actors = ['subject1', 'subject2', 'subject3']
            act_num = '123'
        elif val:
            self.actors = ['subject4'] ## TODO: This could be modified. Check if the amount of data of subject 4 is enough.
            act_num = '4'
        else:
            self.actors = ['subject5']
            act_num = '5'
        

        filename = './pickle_data/lafan1_data_actors_' + act_num + '_win_' + str(self.window) + '_off_' + str(self.offset) + '.pkl'

        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                self.X, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map = pickle.load(f)
                print('Dataset loaded! (', filename, ')')
        else:
            # Load the dataset using the existing get_lafan1_set function --> ATTENTION! WE KEEP THE GLOBAL Q. FOR THE X WE DON'T CARE, SINCE THE ROOT IS ALWAYS GLOBAL
            self.X, _, _, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map = get_lafan1_set(self.data_dir, self.actors, self.window, self.offset) ### OJO!!!!! TODO: HE CAMBIADO ESTO PARA QUE LA X ESTÉ CENTRADA

            # with open(filename, 'wb') as f:
            with open(filename, 'wb') as f:
                pickle.dump((self.X, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map), f)
                print('Dataset saved as ', filename)

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

        sample = {
            'X': torch.tensor(X, dtype=torch.float32),
            'Q': torch.tensor(Q, dtype=torch.float32),
            'parents': torch.tensor(parents, dtype=torch.int64),
            'contacts_l': torch.tensor(contacts_l, dtype=torch.bool),
            'contacts_r': torch.tensor(contacts_r, dtype=torch.bool)
        }

        return sample
    
    
class Lafan1DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, window: int, offset: int):
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

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
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
            self.train_dataset = Lafan1Dataset(self.data_dir, window=self.window, offset=self.offset, train=True, val=False)
            self.val_dataset = Lafan1Dataset(self.data_dir, window=self.window, offset=self.offset, train=False, val=True)

        elif stage == 'test':
            self.test_dataset = Lafan1Dataset(self.data_dir, window=self.window, offset=self.offset, train=False, val=False)
    
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
