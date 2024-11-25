
import numpy as np
from inbetweening.data_processing.process_data import Lafan1DataModule, Lafan1Dataset
from inbetweening.evaluation.baseline import full_interpolation


def main(folder_path, samples_to_generate, gap_sizes, window, offset, interpolate_X_and_Q):
    bvh_path = "../../data"  # Update this with the actual path
    
    # Test by retrieving a sample from the dataset
    data_module = Lafan1DataModule(
        data_dir=bvh_path,
        batch_size=1,
        window=window,
        offset=offset,
    )

    data_module.setup(stage='test')

    for gap_size in gap_sizes:
        for i in range(samples_to_generate):
            sample_index = i * 50  # Adjust this index as needed
            test_dataset = data_module.test_dataset
            sample = test_dataset[sample_index]
            sample = {key: value.cpu() for key, value in sample.items()}

            n_frames = sample['Q'].shape[0]
            start_frame = int((n_frames - gap_size) / 2)
            masked_frames = list(range(start_frame, start_frame + gap_size))

            mask = np.zeros(n_frames, dtype=int)
            mask[masked_frames] = 1

            path_interpolated_bvh = folder_path + f'/interp_output_{sample_index}_original.bvh'
            path_gt_bvh = folder_path + f'/interp_output_{sample_index}_generated_{gap_size}_fr.bvh'

            full_interpolation(sample, masked_frames, path_interpolated_bvh, path_gt_bvh, interpolate_X_and_Q=interpolate_X_and_Q)

if __name__ == "__main__":
    folder_path = '/proj/diffusion-inbetweening/inbetweening/evaluation/generated_interpolation_X_Q'
    samples_to_generate = 51
    gap_sizes = [5, 10, 15]
    window = 50
    offset = 20
    interpolate_X_and_Q = True
    main(folder_path, samples_to_generate, gap_sizes, window, offset, interpolate_X_and_Q)