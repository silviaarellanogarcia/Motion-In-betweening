import numpy as np
from scipy.spatial.transform import Rotation as R

from inbetweening.data_processing.extract import bvh_to_item
from inbetweening.data_processing.utils import quat_fk
from inbetweening.utils.aux_functions import plot_3d_skeleton_with_lines


def quaternion_rotate(quaternion, vector):
    """
    Apply a rotation (given as a quaternion) to a vector.
    """
    # Create a rotation object from the quaternion (assuming [x, y, z, w] order)
    r = R.from_quat(quaternion)
    # Apply the rotation to the vector
    rotated_vector = r.apply(vector)
    return rotated_vector


def compute_L2Q(Q_gt_global, Q_preds_global, masks):
    """
    Computes Global quaternion loss (L2Q)

    Inputs:
        Q_gt: Array with the global quaternions for each joint in the ground_truth sequences
        Q_pred: Array with the global quaternions for each joint in the predicted sequences
        masks: Array with 1s if that sample was newly generated and 0 if it was kept.
    
    Outputs:
        l2q (int): Global quaternion loss
    """
    l2q = 0
    D = len(Q_gt_global) # Length of the test dataset
    T = len(Q_gt_global[0][masks[0] == 1]) # Transition length (all the tests must have the same transition length)

    for d in range(D):
        Q_gt_transition_frames = Q_gt_global[d][masks[d] == 1]
        Q_preds_transition_frames = Q_preds_global[d][masks[d] == 1]

        for t in range(T):
            l2q += np.linalg.norm(Q_gt_transition_frames[t] - Q_preds_transition_frames[t])

    l2q = l2q/(D*T)

    return l2q

# def compute_L2P(X_gt_global, X_preds_global, masks, parents):
#     """
#     Computes Global position loss (L2P).

#     Inputs:
#         X_gt: Array with the local position for each joint in the ground_truth sequences
#         X_pred: Array with the local position for each joint in the predicted sequences
#         masks: Array with 1s if that sample was newly generated and 0 if it was kept.
#         Q_global_gt: Global quaternion rotation of the ground truth sequence
#         Q_global_pred: Global quaternion rotation of the prediction sequence
    
#     Outputs:
#         l2p (int): Global position loss
#     """
#     l2p = 0
#     D = len(X_gt) # Length of the test dataset
#     T = len(X_gt[0][masks[0] == 1]) # Transition length (all the tests must have the same transition length)
#     count = 0

#     # plot_3d_skeleton_with_lines(X_gt_global, parents, sequence_index=0, frames_range=(10, 15))
#     # plot_3d_skeleton_with_lines(X_preds_global, parents, sequence_index=0, frames_range=(10, 15))
#     mean_train = np.load('./mean_train.npy')
#     std_train = np.load('./std_train.npy')

#     for d in range(D):
#         X_gt_transition_frames = X_gt_global[d][masks[d] == 1]
#         X_preds_transition_frames = X_preds_global[d][masks[d] == 1]

#         for t in range(T):
#             X_gt_norm = (X_gt_transition_frames[t] - mean_train) / std_train
#             X_pred_norm = (X_preds_transition_frames[t] - mean_train) / std_train

#             l2p += np.linalg.norm(X_gt_norm - X_pred_norm)

#     l2p /= (D*T)

#     return l2p


def compute_L2P(X_gt_global, X_preds_global, masks):
    """
    Computes Global position loss (L2P) with updated mean and std dev shapes.

    Inputs:
        X_gt_global: Array with the global positions for each joint in the ground truth sequences (T, J, 3)
        X_preds_global: Array with the global positions for each joint in the predicted sequences (T, J, 3)
        masks: Array with 1s if that sample was newly generated and 0 if it was kept.
    
    Outputs:
        l2p (float): Global position loss
    """
    accumulated = 0.0  # To accumulate the L2P loss
    count = 0  # To count the total number of joints processed

    D = len(X_gt_global)  # Number of sequences in the test dataset

    # plot_3d_skeleton_with_lines(X_gt_global, parents, sequence_index=0, frames_range=(10, 15))
    # plot_3d_skeleton_with_lines(X_preds_global, parents, sequence_index=0, frames_range=(10, 15))

    # Load mean and standard deviation for normalization (Shape: (1, 66, 1))
    mean_train = np.load('./mean_train2.npy')  # Shape: (1, 66, 1)
    std_train = np.load('./std_train2.npy')    # Shape: (1, 66, 1)

    for d in range(D):
        # Select transition frames for the current sequence
        X_gt_transition_frames = X_gt_global[d][masks[d] == 1]  # Shape: (frames, joints, 3)
        X_preds_transition_frames = X_preds_global[d][masks[d] == 1]  # Shape: (frames, joints, 3)

        # Flatten the positions (reshape from (frames, joints, 3) to (T, 66)) --> There are 22 joints, so 22*3 = 66
        X_gt_transition_flat = X_gt_transition_frames.reshape(X_gt_transition_frames.shape[0], -1)  # Shape: (T, 66)
        X_preds_transition_flat = X_preds_transition_frames.reshape(X_preds_transition_frames.shape[0], -1)  # Shape: (T, 66)

        # Normalize --> Even if it is not mentioned in all papers, some claim that everyone applies it
        X_gt_norm = (X_gt_transition_flat - mean_train.squeeze(-1)) / std_train.squeeze(-1)
        X_preds_norm = (X_preds_transition_flat - mean_train.squeeze(-1)) / std_train.squeeze(-1)

        # Compute L2 norm per frame and accumulate (sum over joints)
        accumulated += np.linalg.norm(X_gt_norm - X_preds_norm, axis=1).sum()

        # Update count with the total number of joints processed
        count += X_gt_transition_flat.shape[0] * X_gt_transition_flat.shape[1]  # Number of frames * number of joints (66)

    # Compute the average loss over all sequences and frames
    l2p = accumulated / count

    return l2p


def fast_npss(gt_seq, pred_seq):
    # Obtained from the LAFAN1 repository
    """
    Computes Normalized Power Spectrum Similarity (NPSS).

    This is the metric proposed by Gropalakrishnan et al (2019).
    This implementation uses numpy parallelism for improved performance.

    :param gt_seq: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
    :param pred_seq: shape : (Batchsize, Timesteps, Dimension)
    :return: The average npss metric for the batch
    """
    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))

    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)

    # Normalize powers with totals
    gt_norm_power = gt_power / gt_total_power[:, np.newaxis, :]
    pred_norm_power = pred_power / pred_total_power[:, np.newaxis, :]

    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    power_weighted_emd = np.average(emd, weights=gt_total_power)

    return power_weighted_emd


if __name__ == "__main__":
    gt_path = './../model/output9_original.bvh'
    preds_path = './../model/output9_denoised_15_fr.bvh'

    n_frames = 50
    offset = 20
    gap_size = 15

    X_gt, Q_gt, X_gt_global, Q_gt_global, parents, _, _, _ = bvh_to_item(gt_path, window=n_frames, offset=offset)
    X_pred, Q_pred, X_pred_global, Q_pred_global, parents, _, _, _ = bvh_to_item(preds_path, window=n_frames, offset=offset)

    start_frame = int((n_frames - gap_size) / 2)
    masked_frames = list(range(start_frame, start_frame + gap_size))

    mask = np.zeros(n_frames, dtype=int)
    mask[masked_frames] = 1
    mask = np.expand_dims(mask, axis=0)

    if Q_pred.shape[0] < Q_gt.shape[0]: ## Keep only the sequence that has been interpolated
        Q_gt = Q_gt[:Q_pred.shape[0]]
        X_gt = X_gt[:X_pred.shape[0]]

    Q_gt_global, X_gt_global = quat_fk(Q_gt, X_gt, parents)
    Q_pred_global, X_pred_global = quat_fk(Q_pred, X_pred, parents)

    l2q = compute_L2Q(Q_gt_global, Q_pred_global, mask)
    print("L2Q: ", l2q)
    l2p = compute_L2P(X_gt_global, X_pred_global, mask)
    print("L2P: ", l2p)

    npss = fast_npss(X_gt, X_pred)
    print("NPSS: ", npss)
