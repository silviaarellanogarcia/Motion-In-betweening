import numpy as np
import torch
from inbetweening.data_processing.process_data import Lafan1DataModule, Lafan1Dataset
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pymotion.rotations.ortho6d as sixd

from inbetweening.utils.convert_to_bvh import write_bvh

def interpolate_position(X, masked_frames):
    num_gap_frames = len(masked_frames)

    # Initialize an array to hold the new interpolated positions
    new_seq_X = X.clone()
    new_seq_X[masked_frames, 0, :] = 0 ## Set all the offsets. Now, only the hip movements are missing

    last_positions = X[masked_frames[0] - 1, 0, :]
    target_positions = X[masked_frames[-1] + 1, 0, :]

    for frame in range(num_gap_frames):
        # Calculate the interpolation factor
        alpha = (frame + 1) / (num_gap_frames + 1)  # Normalized factor between 0 and 1
        
        # Linearly interpolate between the last position and the target sample
        new_pos = (1 - alpha) * last_positions + alpha * target_positions
        
        # Store the new position in the new sequence
        new_seq_X[masked_frames[0] + frame, 0, :] = new_pos

    return new_seq_X

def interpolate_quaternions(Q, masked_frames):
    num_joints = Q.shape[1]
    start_frame = masked_frames[0] - 1
    target_frame = masked_frames[-1] + 1
    num_gap_frames = len(masked_frames)
    
    for joint in range(num_joints):
        # Get quaternions for the start and end frames for the current joint
        quat_start = Q[start_frame, joint, :]
        quat_end = Q[target_frame, joint, :]
        
        # The SLERP interpolator works with the Rotation class!
        # Create the rotations and the SLERP interpolator
        rotations = R.from_quat([quat_start, quat_end])
        slerp = Slerp([start_frame, target_frame], rotations)
        
        gap_times = np.linspace(start_frame, target_frame, num_gap_frames + 2)[1:-1] # Instant where to fit each gap frame
        
        # Perform SLERP interpolation for the gap frames
        interpolated_rots = slerp(gap_times)
        
        gap_frames = range(start_frame + 1, target_frame)
        # Replace the missing frames in Q with the interpolated quaternions
        for idx, frame in enumerate(gap_frames):
            Q[frame, joint, :] = torch.from_numpy(interpolated_rots[idx].as_quat())

    return Q

def full_interpolation(sample, masked_frames, path_interpolated_bvh, path_gt_bvh = None, interpolate_X = True, interpolate_Q = True):
    '''
    Interpolate rotations and positions. The input rotation is in Ortho6D, but it is converted to quaternions, and then interpolated.
    If interpolate_X_and_Q is False, it only interpolates Q, and uses the real X.
    '''
    X = sample['X']
    Q = sample['Q'].reshape(sample['Q'].shape[0], sample['Q'].shape[1], 3, 2) # Shape (frames, joints, 6)
    Q = Q.numpy()
    Q = sixd.to_quat(Q) # Shape (frames, joints, 3, 2)
    Q = torch.from_numpy(Q).float()
    
    if path_gt_bvh is not None:
        write_bvh(path_gt_bvh, X, sample['parents'], Q_global=Q)

    if interpolate_X and not interpolate_Q:
        X[masked_frames, 0, :] = 0
        interpolated_X = interpolate_position(X, masked_frames)
        write_bvh(path_interpolated_bvh, interpolated_X, sample['parents'], Q_global=Q)

    Q[masked_frames, :, :] = 0
    interpolated_Q = interpolate_quaternions(Q, masked_frames)

    if interpolate_X and interpolate_Q:
        X[masked_frames, 0, :] = 0
        interpolated_X = interpolate_position(X, masked_frames)
        write_bvh(path_interpolated_bvh, interpolated_X, sample['parents'], Q_global=interpolated_Q)
    elif interpolate_Q:
        interpolated_X = None
        write_bvh(path_interpolated_bvh, X, sample['parents'], Q_global=interpolated_Q)

    return interpolated_X, interpolated_Q


if __name__ == "__main__":
    bvh_path = "../../data"  # Update this with the actual path
    dataset = Lafan1Dataset(bvh_path, window=50, offset=20, train=True)
    gap_size = 15
    window = 50
    offset = 20

    # Test by retrieving a sample from the dataset
    data_module = Lafan1DataModule(
        data_dir=bvh_path,
        batch_size=1,
        window=window,
        offset=offset,
    )

    data_module.setup(stage='test')

    sample_index = 10  # Adjust this index as needed
    test_dataset = data_module.test_dataset
    sample = test_dataset[sample_index]
    sample = {key: value.cpu() for key, value in sample.items()}

    n_frames = sample['Q'].shape[0]
    start_frame = int((n_frames - gap_size) / 2)
    masked_frames = list(range(start_frame, start_frame + gap_size))

    mask = np.zeros(n_frames, dtype=int)
    mask[masked_frames] = 1

    path_interpolated_bvh = 'output_walk_int2.bvh'
    path_gt_bvh = 'output_walk_gt2.bvh'

    full_interpolation(sample, masked_frames, path_interpolated_bvh, path_gt_bvh, interpolate_X_and_Q=False)