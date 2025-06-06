## This file belongs to the ubisoft-laforge-animation-dataset repository

import re, os, ntpath
import numpy as np
import inbetweening.data_processing.utils as utils
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines, plot_root

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1 ## Helps to keep track of the hierarchy
    end_site = False ## True when the last joint of a sequence of bones is reached (ex. reaching the tip of the finger)

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)

def process_window(start, end, seq_name, anim, X, Q, X_global, Q_global, contacts_l, contacts_r, seq_names):
    """
    Process a single sliding window of data and append results to the provided lists.

    :param start: Start index of the window
    :param end: End index of the window
    :param seq_name: Name of the sequence
    :param anim: Animation object containing positions, quaternions, and parents
    :param X, Q, X_global, Q_global, contacts_l, contacts_r, seq_names: Lists to append the processed data
    """
    q, x = utils.quat_fk(anim.quats[start:end], anim.pos[start:end], anim.parents)
    c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
    X.append(anim.pos[start:end])
    Q.append(anim.quats[start:end])
    X_global.append(q)
    Q_global.append(x)
    contacts_l.append(c_l)
    contacts_r.append(c_r)
    seq_names.append(seq_name)

def bvh_to_item(bvh_path, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps) --> windows overlap a bit between each other
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    seq_names = []
    X = []
    Q = []
    X_global = []
    Q_global = []
    contacts_l = []
    contacts_r = []
    index_map = []  # Stores the start and end index for each bvh file

    # Extract
    global_idx = 0

    if bvh_path.endswith('.bvh'):
        seq_name = ntpath.basename(bvh_path[:-4])

        # print('Processing file {}'.format(os.path.basename(bvh_path)))
        anim = read_bvh(bvh_path)

        # Sliding windows
        i = 0
        start_idx = global_idx

        if window == anim.pos.shape[0]:
            process_window(0, window, seq_name, anim, X, Q, X_global, Q_global, contacts_l, contacts_r, seq_names)
            global_idx += 1

        while i+window < anim.pos.shape[0]:
            process_window(i, i + window, seq_name, anim, X, Q, X_global, Q_global, contacts_l, contacts_r, seq_names)
            i += offset
            global_idx += 1

        end_idx = global_idx  # Store the end index for this file
        index_map.append((start_idx, end_idx - 1, os.path.basename(bvh_path)))  # Store the range of indices and the file name

    X = np.asarray(X)
    Q = np.asarray(Q)
    X_global = np.asarray(X_global)
    Q_global = np.asarray(Q_global)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0 --> Center the sequences around the XZ plane
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame --> We always want to have the first pose facing "us", so we change all the other poses accordding to this.
    # Shape (n_sequences, window_size, n_joints, n_dimensions) --> n_dimensions is always 3, x, y, z
    X, Q, X_gobal_new, Q_global_new = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, X_gobal_new, Q_global_new, anim.parents, contacts_l, contacts_r, index_map


def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps) --> windows overlap a bit between each other
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []
    index_map = []  # Stores the start and end index for each bvh file

    # Extract
    bvh_files = os.listdir(bvh_path)
    global_idx = 0

    for file in bvh_files:
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                # print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # Sliding windows
                i = 0
                start_idx = global_idx

                while i+window < anim.pos.shape[0]:
                    q, x = utils.quat_fk(anim.quats[i: i+window], anim.pos[i: i+window], anim.parents) ## Returns the orientation and global positions
                    # Extract contacts --> c_l and c_r are 2Dd arrays because the contact is evaluated at 2 different joints (ex. foot and heel)
                    c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    seq_names.append(seq_name)
                    subjects.append(subject)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset
                    global_idx += 1

                end_idx = global_idx  # Store the end index for this file
                index_map.append((start_idx, end_idx - 1, file))  # Store the range of indices and the file name

    X = np.asarray(X)
    Q = np.asarray(Q)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0 --> Center the sequences around the XZ plane
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame --> We always want to have the first pose facing "us", so we change all the other poses accordding to this.
    # Shape (n_sequences, window_size, n_joints, n_dimensions) --> n_dimensions is always 3, x, y, z
    X, Q , Q_global_new, X_global_new = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)
    # plot_3d_skeleton_with_lines(X_global_new, anim.parents, sequence_index=0, frames_range=(0, 2))
    # plot_root(X_global_new[:, :, 0, :], start_frame=0, end_frame=49, sequence_index=0)

    return X, Q, X_global_new, Q_global_new, anim.parents, contacts_l, contacts_r, index_map


def get_train_stats(bvh_folder, train_set):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    print('Building the train set...')
    xtrain, qtrain, _, _, parents, _, _, _ = get_lafan1_set(bvh_folder, train_set, window=50, offset=20)

    print('Computing stats...\n')
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)

    return x_mean, x_std, offsets

if __name__ == "__main__":
    bvh_path = "/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data1"
    actors = ['subject1', 'subject2', 'subject3', 'subject4']
    X, Q, _, _, parents, contacts_l, contacts_r, index_map = get_lafan1_set(bvh_path, actors, window=50, offset=20)
    print(X)

    x_mean, x_std, _ = get_train_stats(bvh_path, actors)
    # np.save('mean_train.npy', x_mean)
    # np.save('std_train.npy', x_std)
    print(x_std)