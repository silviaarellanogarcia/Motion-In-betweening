import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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