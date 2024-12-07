
import numpy as np
from inbetweening.data_processing.process_data import Lafan1DataModule, Lafan1Dataset
from inbetweening.evaluation.baseline import full_interpolation


def main(folder_path, gap_sizes, window, offset, interpolate_X, interpolate_Q, samples_to_generate=None):
    bvh_path = "../../data"  # Update this with the actual path
    
    # Test by retrieving a sample from the dataset
    data_module = Lafan1DataModule(
        data_dir=bvh_path,
        batch_size=1,
        window=window,
        offset=offset,
    )

    data_module.setup(stage='test') ## TODO: CHANGE!
    test_dataset = data_module.test_dataset

    if samples_to_generate == None:
        samples_to_generate = len(test_dataset)

    for gap_size in gap_sizes:
        for i in range(samples_to_generate):
            sample_index = i  # Adjust this index as needed
            sample = test_dataset[sample_index]
            sample = {key: value.cpu() for key, value in sample.items()}

            n_frames = sample['Q'].shape[0]
            start_frame = int((n_frames - gap_size) / 2)
            masked_frames = list(range(start_frame, start_frame + gap_size))

            mask = np.zeros(n_frames, dtype=int)
            mask[masked_frames] = 1

            path_gt_bvh = folder_path + f'/interp_output_{sample_index}_original.bvh'
            path_interpolated_bvh = folder_path + f'/interp_output_{sample_index}_generated_{gap_size}_fr.bvh'

            full_interpolation(sample, masked_frames, path_interpolated_bvh, path_gt_bvh, interpolate_X = interpolate_X, interpolate_Q = interpolate_Q)

if __name__ == "__main__":
    folder_path = '/proj/diffusion-inbetweening/inbetweening/evaluation/generated_interpolation_only_X_test_set'
    samples_to_generate = None
    gap_sizes = [5, 10, 15]
    window = 50
    offset = 20
    interpolate_X = True
    interpolate_Q = False
    main(folder_path, gap_sizes, window, offset, interpolate_X, interpolate_Q, samples_to_generate)