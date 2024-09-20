import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from inbetweening.data_processing.extract import get_lafan1_set

class Lafan1Dataset(Dataset):
    """LLFAN1 Dataset class."""

    def __init__(self, data_dir, window=50, offset=20, train=True):
        """
        Args:
            data_dir (string): Directory with the dataset.
            window (int): Size of the sliding window
            offset (int): Offset between windows (in timesteps)
            train (boolean): Indicates if the dataset is for training or testing

        Outputs:
            X: local positions
            Q: local quaternions
            parents: list of parent indices defining the bone hierarchy
            contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
            contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
        """

        self.data_dir = data_dir
        self.window = window
        self.offset = offset

        if train:
            self.actors = ['subject1', 'subject2', 'subject3', 'subject4']
        else:
            self.actors = ['subject5']
        
        # Load the dataset using the existing get_lafan1_set function
        self.X, self.Q, self.parents, self.contacts_l, self.contacts_r, self.index_map = get_lafan1_set(self.data_dir, self.actors, self.window, self.offset)

    def __len__(self):
        return len(self.X) ## Returns the number of sequences.


    def __getitem__(self, idx):
        ### TODO: TEST THIS
        """
        Args:
            idx: Inddex of the sample to retrieve
        """
        X = self.X[idx]
        Q = self.Q[idx]
        parents = self.parents
        contacts_l = self.contacts_l[self.index_map[idx][0] : self.index_map[idx][1]]
        contacts_r = self.contacts_r[self.index_map[idx][0] : self.index_map[idx][1]]

        sample = {
            'X': X,
            'Q': Q,
            'parents': parents,
            'contacts_l': contacts_l,
            'contacts_r': contacts_r
        }

        return sample
    
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
