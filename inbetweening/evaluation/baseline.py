import numpy as np
from inbetweening.data_processing.process_data import Lafan1Dataset


def interpolate_position_and_angles(past_context_seq, target_sample, num_frames_to_interpolate):
    len_past_seq = len(past_context_seq)
    idx_frame_to_generate = len_past_seq + 1 ## This index starts with 1 to be able to compute the weights correctly
    idx_target = len_past_seq + num_frames_to_interpolate + 1
    new_seq = []
    new_pos = np.zeros((past_context_seq.shape[1], past_context_seq.shape[2]))


    for frame in range(1, num_frames_to_interpolate):
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

        # Continue interpolating the quaternions

        # Recompute the contacts ????

        # Keep the parents the same
    
    new_seq = np.array(new_seq)
    # Concatenate past_context_seq, new_seq, and target_sample
    final_seq = np.concatenate((past_context_seq, new_seq, target_sample[np.newaxis, :]), axis=0)

    return final_seq

if __name__ == "__main__":
    bvh_path = bvh_path = "/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/data1"  # Update this with the actual path
    dataset = Lafan1Dataset(bvh_path, window=50, offset=20, train=True)

    # Test by retrieving a sample from the dataset
    sample_idx = 0  # Test with the first sample
    sample = dataset[sample_idx]

    past_context_seq = sample['X'][:10, :, :]
    target_sample = sample['X'][14, :, :]
    num_frames_to_interpolate = 5

    new_pos = interpolate_position_and_angles(past_context_seq, target_sample, num_frames_to_interpolate)
