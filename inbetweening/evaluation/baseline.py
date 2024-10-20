import numpy as np
import torch
from inbetweening.data_processing.process_data import Lafan1Dataset
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from inbetweening.utils.convert_to_bvh import write_bvh


def interpolate_position(past_context_seq, target_sample, num_gap_frames):
    last_position = past_context_seq[-1]

    # Initialize an array to hold the new interpolated positions
    new_seq = np.zeros((num_gap_frames, last_position.shape[0], last_position.shape[1]))
    new_seq[:, 1:, :] = last_position[1:, :] ## Set all the offsets. Now, only the hip movements are missing

    for frame in range(num_gap_frames):
        # Calculate the interpolation factor
        alpha = (frame + 1) / (num_gap_frames + 1)  # Normalized factor between 0 and 1
        
        # Linearly interpolate between the last position and the target sample
        new_pos = (1 - alpha) * last_position[0, :] + alpha * target_sample[0, :]
        
        # Store the new position in the new sequence
        new_seq[frame, 0] = new_pos

    # Concatenate past_context_seq, new_seq, and target_sample
    final_seq = np.concatenate((past_context_seq, new_seq, target_sample[np.newaxis, :]), axis=0)

    return final_seq


def interpolate_quaternions(Q, start_frame, target_frame, num_gap_frames):
    num_joints = Q.shape[1]
    
    for joint in range(num_joints):
        # Get quaternions for the start and end frames for the current joint
        quat_start = Q[start_frame, joint]
        quat_end = Q[target_frame, joint]
        
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
            Q[frame, joint] = torch.from_numpy(interpolated_rots[idx].as_quat())

    return Q


if __name__ == "__main__":
    bvh_path = "/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data_walking"  # Update this with the actual path
    dataset = Lafan1Dataset(bvh_path, window=50, offset=20, train=True)

    # Test by retrieving a sample from the dataset
    sample_idx = 0  # Test with the first sample
    sample = dataset[sample_idx]
    
    write_bvh('output_walk_gt2.bvh', sample['X'], sample['parents'], Q_global=sample['Q'])

    past_context_seq = sample['X'][:10, :, :]
    target_sample = sample['X'][15, :, :]
    num_gap_frames = 5

    interpolated_X = interpolate_position(past_context_seq, target_sample, num_gap_frames)
    interpolated_Q = interpolate_quaternions(sample['Q'], 9, 15, 5)

    write_bvh('output_walk_int2.bvh', interpolated_X, sample['parents'], Q_global=interpolated_Q[:16, :, :])