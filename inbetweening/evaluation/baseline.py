import numpy as np
from inbetweening.data_processing.process_data import Lafan1Dataset
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def interpolate_position(past_context_seq, target_sample, num_gap_frames):
    len_past_seq = len(past_context_seq)
    idx_frame_to_generate = len_past_seq + 1 ## This index starts with 1 to be able to compute the weights correctly
    idx_target = len_past_seq + num_gap_frames + 1
    new_seq = []
    new_pos = np.zeros((past_context_seq.shape[1], past_context_seq.shape[2]))


    for frame in range(1, num_gap_frames):
        normalizing_factor = 0
        # First interpolate the position
        for i in range(past_context_seq.shape[0]):
            weight = 1 - (idx_frame_to_generate - i)/idx_target
            normalizing_factor += weight
            new_pos += weight * past_context_seq[i - 1] # Put ['X'] if I'm passing the whole seq and not only the pos
        
        weight = 1 - (idx_target - idx_frame_to_generate)/idx_target
        normalizing_factor += weight
        new_pos += weight * target_sample
        new_pos /= normalizing_factor

        new_seq.append(new_pos)
    
    new_seq = np.array(new_seq)
    # Concatenate past_context_seq, new_seq, and target_sample
    final_seq = np.concatenate((past_context_seq, new_seq, target_sample[np.newaxis, :]), axis=0)

    return final_seq


def interpolate_quaternions(Q, start_frame, target_frame, num_gap_frames):
    num_joints = Q.shape[1]
    
    for joint in range(num_joints):
        # Get quaternions for the start and end frames for the current joint
        quat_start = Q[start_frame, joint]
        quat_end = Q[target_frame, joint]
        
        # The SLERP interpolator works with the Rotation class. 
        # Create the rotations and the SLERP interpolator
        rotations = R.from_quat([quat_start, quat_end])
        slerp = Slerp([start_frame, target_frame], rotations)
        
        gap_times = np.linspace(start_frame, target_frame, num_gap_frames + 2)[1:-1] # Instant where to fit each gap frame
        
        # Perform SLERP interpolation for the gap frames
        interpolated_rots = slerp(gap_times)
        
        gap_frames = range(start_frame + 1, target_frame)
        # Replace the missing frames in Q with the interpolated quaternions
        for idx, frame in enumerate(gap_frames):
            Q[frame, joint] = interpolated_rots[idx].as_quat()

    return Q


if __name__ == "__main__":
    bvh_path = bvh_path = "/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data1"  # Update this with the actual path
    dataset = Lafan1Dataset(bvh_path, window=50, offset=20, train=True)

    # Test by retrieving a sample from the dataset
    sample_idx = 0  # Test with the first sample
    sample = dataset[sample_idx]

    past_context_seq = sample['X'][:10, :, :]
    target_sample = sample['X'][14, :, :]
    num_gap_frames = 5

    interpolated_X = interpolate_position(past_context_seq, target_sample, num_gap_frames)
    interpolated_Q = interpolate_quaternions(sample['Q'], 9, 15, 5)
