import csv
import os
import numpy as np
import pandas as pd
from inbetweening.data_processing.extract import bvh_to_item
from inbetweening.data_processing.utils import quat_fk
from inbetweening.evaluation.metrics import compute_L2P, compute_L2Q, fast_npss

def compute_means_by_num_frames(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Group by the "num_frames" column and compute the mean for "L2Q", "L2P", "NPSS"
    grouped_means = df.groupby("num_frames")[["L2Q", "L2P", "NPSS"]].mean()

    # Print the grouped means
    print("Mean Values Grouped by num_frames:")
    print(grouped_means)

    return grouped_means

def get_sample_nums(folder_path):
    sample_nums = set()  # To store unique sample numbers

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".bvh"):
            # Split the filename into parts
            parts = filename.split('_')
            if len(parts) > 2 and parts[0] == "diff" and parts[1] == "output":
                sample_num = parts[2]  # Extract the sample number
                sample_nums.add(sample_num)

    return sample_nums

def pair_denoised_with_original(folder_path):
    file_dict = {}  # To group files by sample_num
    paired_dict = {}  # To store pairs of denoised and original files

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".bvh") and filename.startswith("diff_output_"):
            # Split the filename to extract components
            parts = filename.split('_')
            if len(parts) > 2:
                sample_num = parts[2]  # Extract sample_num
                file_type = "denoised" if "denoised" in filename else "original"
                
                # Group files by sample_num
                if sample_num not in file_dict:
                    file_dict[sample_num] = {"denoised": [], "original": []}
                
                # Append the filename to the correct list
                file_dict[sample_num][file_type].append(filename)

    # Pair denoised and original files
    for sample_num, files in file_dict.items():
        denoised_files = files.get("denoised", [])
        original_files = files.get("original", [])
        
        # For each denoised file, pair it with an original file (if available)
        for denoised_file in denoised_files:
            if original_files:
                paired_dict[denoised_file] = original_files[0]  # Assuming all share the same original

    return paired_dict


def main(folder_path, n_frames, offset, output_csv_path):
    pairs = pair_denoised_with_original(folder_path)

    # Prepare results to save
    results = []

    for denoised, original in pairs.items():
        gt_path = os.path.join(folder_path, original)
        preds_path = os.path.join(folder_path, denoised)

        num_frames = int(denoised.split('_')[4])  # Extract num_frames from the filename
        sample_num = denoised.split('_')[2]  # Extract sample_num

        X_gt, Q_gt, X_gt_global, Q_gt_global, parents, _, _, _ = bvh_to_item(gt_path, window=n_frames, offset=offset)
        X_pred, Q_pred, X_pred_global, Q_pred_global, parents, _, _, _ = bvh_to_item(preds_path, window=n_frames, offset=offset)

        start_frame = int((n_frames - num_frames) / 2)
        masked_frames = list(range(start_frame, start_frame + num_frames))

        mask = np.zeros(n_frames, dtype=int)
        mask[masked_frames] = 1
        mask = np.expand_dims(mask, axis=0)

        if Q_pred.shape[0] < Q_gt.shape[0]:  # Keep only the sequence that has been interpolated
            Q_gt = Q_gt[:Q_pred.shape[0]]
            X_gt = X_gt[:X_pred.shape[0]]

        Q_gt_global, X_gt_global = quat_fk(Q_gt, X_gt, parents)
        Q_pred_global, X_pred_global = quat_fk(Q_pred, X_pred, parents)

        l2q = compute_L2Q(Q_gt_global, Q_pred_global, mask)
        l2p = compute_L2P(X_gt_global, X_pred_global, mask)
        npss = fast_npss(X_gt, X_pred)

        print(f"Sample: {sample_num}, Frames: {num_frames}, L2Q: {l2q}, L2P: {l2p}, NPSS: {npss}")

        # Append results for CSV
        results.append([sample_num, num_frames, l2q, l2p, npss])

    # Save results to a CSV file
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_num", "num_frames", "L2Q", "L2P", "NPSS"])  # Write header
        writer.writerows(results)  # Write data rows

    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    folder_path = "/proj/diffusion-inbetweening/inbetweening/model/generated_samples"  # Replace with your folder path
    output_csv_path = "/proj/diffusion-inbetweening/inbetweening/model/OUTPUT.csv"
    n_frames = 50
    offset = 20

    main(folder_path, n_frames, offset, output_csv_path)
    compute_means_by_num_frames(output_csv_path)