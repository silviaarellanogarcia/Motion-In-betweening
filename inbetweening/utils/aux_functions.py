import os
import matplotlib.pyplot as plt

def plot_3d_skeleton_with_lines(X_gt_global, hierarchy, sequence_index=0, frames_range=None):
    """
    Plots the global positions of all joints in 3D for each frame and connects each joint to its parent with a line.

    Args:
        X_gt_global: Array of global positions (sequences, frames, joints, 3).
        hierarchy: Array that specifies the parent index of each joint.
        sequence_index: Index of the sequence to plot (default is 0).
        frames_range: Tuple specifying the range of frames to plot (start, end).
                      If None, it plots all frames.
    """
    # Extract the sequence data for the given sequence index
    sequence = X_gt_global[sequence_index]

    # If frames_range is None, plot all frames
    if frames_range is None:
        frames_range = (0, len(sequence))
    
    start_frame, end_frame = frames_range

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over the frames in the specified range
    for t in range(start_frame, end_frame):
        joints = sequence[t]  # shape (num_joints, 3)

        # Extract x, y, z coordinates
        xs = joints[:, 0]
        ys = joints[:, 1]
        zs = joints[:, 2]

        # Plot the joints themselves
        ax.scatter(xs, ys, zs, color='blue')

        # For each joint, plot a line to its parent
        for j in range(1, len(hierarchy)):  # Start at 1 because the root has no parent
            parent = hierarchy[j]
            
            # Draw a line between the current joint and its parent
            ax.plot(
                [xs[j], xs[parent]],  # X coordinates
                [ys[j], ys[parent]],  # Y coordinates
                [zs[j], zs[parent]],  # Z coordinates
                color='black'         # Color of the lines
            )

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Skeleton for Sequence {sequence_index}')

    # Display the plot
    plt.show()

def plot_root(X_root, start_frame, end_frame, sequence_index=0):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sequence = X_root[sequence_index]

    # Iterate over the frames in the specified range
    for t in range(start_frame, end_frame):
        joints = sequence[t]  # shape (num_joints, 3)

        # Extract x, y, z coordinates
        xs = joints[0]
        ys = joints[1]
        zs = joints[2]

        # Plot the joints themselves
        ax.scatter(xs, ys, zs, color='blue')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Root trajectory')
    plt.show()


def plot_root_with_real(X_pred_root, X_real_root, start_frame, end_frame, gap_size, output_dir='plots'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each sequence in the batch
    for sequence_index, (pred_sequence, real_sequence) in enumerate(zip(X_pred_root, X_real_root)):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate over the frames in the specified range
        for t in range(start_frame, end_frame):
            # Predicted root positions
            pred_joints = pred_sequence[t]  # shape (num_joints, 3)
            pred_xs = pred_joints[0]
            pred_ys = pred_joints[1]
            pred_zs = pred_joints[2]

            # Real root positions
            real_joints = real_sequence[t]  # shape (num_joints, 3)
            real_xs = real_joints[0]
            real_ys = real_joints[1]
            real_zs = real_joints[2]

            # Plot the predicted positions
            ax.scatter(pred_xs, pred_ys, pred_zs, color='blue', label='Predicted' if t == start_frame else "")

            # Plot the real positions
            ax.scatter(real_xs, real_ys, real_zs, color='red', label='Reference' if t == start_frame else "")

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Root Trajectory - Sequence {sequence_index}, {gap_size} predicted frames')
        
        # Add a legend
        ax.legend()

        # Save the figure with a unique file name
        save_path = os.path.join(output_dir, f'sequence_{sequence_index}_trajectory_{gap_size}_fr.png')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory

        # print(f"Saved plot for sequence {sequence_index} to {save_path}")
