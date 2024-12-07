import numpy as np
from scipy.spatial.transform import Rotation as R

from inbetweening.data_processing.utils import quat_ik_Q

def write_joint(f, joint_idx, X, parents, level=0):
    """Writes a single joint in the BVH hierarchy."""
    indent = '    ' * level
    if parents[joint_idx] == -1:  # Root joint
        f.write(f"{indent}ROOT Joint{joint_idx}\n")
    else:
        f.write(f"{indent}JOINT Joint{joint_idx}\n")
    
    f.write(f"{indent}{{\n")
    
    # Use the first frame's position as the offset
    offset = X[0, joint_idx]
    f.write(f"{indent}    OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
    
    if parents[joint_idx] == -1:  # Root joint has position and rotation channels
        f.write(f"{indent}    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
    else:  # Other joints only have rotation channels
        f.write(f"{indent}    CHANNELS 3 Zrotation Yrotation Xrotation\n")
    
    # Write child joints recursively
    child_indices = np.where(parents == joint_idx)[0]
    
    if len(child_indices) == 0:  # If no children, add End Site
        f.write(f"{indent}    End Site\n")
        f.write(f"{indent}    {{\n")
        f.write(f"{indent}        OFFSET 0.000000 0.000000 0.000000\n")  # Adjust offset as needed
        f.write(f"{indent}    }}\n")
    else:
        for child in child_indices:
            write_joint(f, child, X, parents, level + 1)
    
    f.write(f"{indent}}}\n")

def write_hierarchy(f, X, parents):
    """Writes the entire joint hierarchy in the BVH file."""
    f.write("HIERARCHY\n")
    write_joint(f, 0, X, parents)  # Start from the root joint (index 0)

def quat_to_euler(q):
    """Converts a quaternion to Euler angles in ZYX order for BVH."""
    return R.from_quat(q).as_euler('ZYX', degrees=True)

def write_motion(f, X, Q, parents, frame_time):
    """Writes the motion section in the BVH file."""
    num_frames, num_joints, _ = X.shape
    f.write("MOTION\n")
    f.write(f"Frames: {num_frames}\n")
    f.write(f"Frame Time: {frame_time:.6f}\n")
    
    # Write motion data for each frame
    for frame in range(num_frames):
        for joint_idx in range(num_joints):
            if parents[joint_idx] == -1:  # Root joint includes position
                pos = X[frame, joint_idx]
                q_reordered = Q[frame, joint_idx][[1, 2, 3, 0]] ###### TODO: I REORDERED THIS
                rot = quat_to_euler(q_reordered)
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} ")
            else:  # Other joints have only rotations
                q_reordered = Q[frame, joint_idx][[1, 2, 3, 0]] ###### T DO: I REORDERED THIS
                rot = quat_to_euler(q_reordered) ## It returns it as ZYX
            f.write(f"{rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f} ")
        f.write("\n")

def write_bvh(filename, X, parents, Q_global=None, Q_local=None, frame_time=1.0 / 30):
    """Main function to write the BVH file."""
    
    if Q_global is None and Q_local is None:
        print("You forgot to provide either Q_local or Q_global!")
        return
    elif Q_local is not None:
        Q = Q_local
    elif Q_global is not None:
        Q = quat_ik_Q(Q_global.detach().cpu().numpy(), parents.cpu())

    with open(filename, 'w') as f:
        # Write the HIERARCHY section
        write_hierarchy(f, X.cpu(), parents.cpu())
        
        # Write the MOTION section
        write_motion(f, X.cpu(), Q, parents.cpu(), frame_time)

    # print("BVH saved!")