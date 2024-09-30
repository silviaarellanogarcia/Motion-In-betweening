import numpy as np
from scipy.spatial.transform import Rotation as R

from inbetweening.data_processing.extract import bvh_to_item

def compute_local_to_global_quat(Q, parents):
    n_sequences, n_frames, n_joints, _ = Q.shape

    Q_global = np.zeros_like(Q)

    for seq in range(n_sequences):
        for frame in range(n_frames):
            for j in range(n_joints):
                if parents[j] == -1: ## This is the root, it's already global.
                    Q_global[seq, frame, j] = Q[seq, frame, j]
                else:
                    parent_global_rot = R.from_quat(Q_global[seq, frame, parents[j]])
                    local_rot = R.from_quat(Q[seq, frame, j])
                    global_rot = parent_global_rot * local_rot

                    Q_global[seq, frame, j] = global_rot.as_quat()
    
    return Q_global


def compute_global_positions(X, Q_global, parents):
    """
    Computes global positions for each frame based on local positions and global rotations.
    
    Args:
    - X: numpy array of shape (n_sequences, n_frames, n_joints, 3) containing local positions.
    - Q_global: numpy array of shape (n_sequences, n_frames, n_joints, 4) containing global rotations as quaternions.
    - parents: numpy array of shape (n_joints,) containing the parent index for each joint.

    Returns:
    - P: numpy array of shape (n_sequences, n_frames, n_joints, 3) containing global positions.
    """
    n_sequences, n_frames, n_joints, _ = X.shape

    X_global = np.zeros((n_sequences, n_frames, n_joints, 3))

    for seq in range(n_sequences):
        for frame in range(n_frames):
            for j in range(n_joints):
                if parents[j] == -1:
                    X_global[seq, frame, j] = X[seq, frame, j] # Root joint: its global position is the same as its local position
                else:
                    parent_position = X_global[seq, frame, parents[j]]
                    joint_rotation = R.from_quat(Q_global[seq, frame, j])
                    rotated_local_position = joint_rotation.apply(X[seq, frame, j]) # Rotate the local position by the joint's global rotation
                    
                    # Compute the global position of the joint based on the parent's position
                    X_global[seq, frame, j] = parent_position + rotated_local_position

    return X_global


def compute_L2Q(Q_gt, Q_pred, masks, parents):
    """
    Computes Global quaternion loss (L2Q) or Global position loss (L2P).

    Inputs:
        Q_gt: Array with the local quaternions for each joint in the ground_truth sequences
        Q_pred: Array with the local quaternions for each joint in the predicted sequences
        masks: Array with 1s if that sample was newly generated and 0 if it was kept.
    
    Outputs:
        l2q (int): Global quaternion loss
    """
    l2q = 0
    D = len(Q_gt) # Length of the test dataset
    T = len(Q_gt[0][masks[0] == 1]) # Transition length (all the tests must have the same transition length)

    Q_gt_global = compute_local_to_global_quat(Q_gt, parents)
    Q_preds_global = compute_local_to_global_quat(Q_pred, parents)

    for d in range(D):
        Q_gt_transition_frames = Q_gt_global[d][masks[d] == 1]
        Q_preds_transition_frames = Q_preds_global[d][masks[d] == 1]

        for t in range(T):
            l2q += np.linalg.norm(Q_gt_transition_frames[t] - Q_preds_transition_frames[t])

    l2q = l2q/(D*T)

    return l2q, Q_gt_global, Q_preds_global


def compute_L2P(X_gt, X_pred, Q_global_gt, Q_global_pred, masks, parents):
    """
    Computes Global position loss (L2P).

    Inputs:
        X_gt: Array with the local position for each joint in the ground_truth sequences
        X_pred: Array with the local position for each joint in the predicted sequences
        masks: Array with 1s if that sample was newly generated and 0 if it was kept.
        Q_global_gt: Global quaternion rotation of the ground truth sequence
        Q_global_pred: Global quaternion rotation of the prediction sequence
    
    Outputs:
        l2p (int): Global position loss
    """
    l2p = 0
    D = len(X_gt) # Length of the test dataset
    T = len(X_gt[0][masks[0] == 1]) # Transition length (all the tests must have the same transition length)

    X_gt_global = compute_global_positions(X_gt, Q_global_gt, parents)
    X_preds_global = compute_global_positions(X_pred, Q_global_pred, parents)

    for d in range(D):
        X_gt_transition_frames = X_gt_global[d][masks[d] == 1]
        X_preds_transition_frames = X_preds_global[d][masks[d] == 1]

        for t in range(T):
            l2p += np.linalg.norm(X_gt_transition_frames[t] - X_preds_transition_frames[t])

    l2p = l2p/(D*T)

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
    gt_path = '/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/Motion-In-betweening/inbetweening/evaluation/output_short.bvh'
    preds_path = '/Users/silviaarellanogarcia/Documents/MSc MACHINE LEARNING/Advanced Project/proyecto/Motion-In-betweening/inbetweening/evaluation/output5.bvh'

    X_gt, Q_gt, parents, _, _, _ = bvh_to_item(gt_path, window=15, offset=15)
    X_pred, Q_pred, parents, _, _, _ = bvh_to_item(preds_path, window=15, offset=15)
    mask = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,0])[np.newaxis, :]
    l2q, Q_gt_global, Q_preds_global = compute_L2Q(Q_gt, Q_pred, mask, parents)
    print("L2Q: ", l2q)
    l2p = compute_L2P(X_gt, X_pred, Q_gt_global, Q_preds_global, mask, parents)
    print("L2P: ", l2p)
    npss = fast_npss(X_gt, X_pred)
    print("NPSS: ", npss)
