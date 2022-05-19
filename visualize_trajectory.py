import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_trajectory(root_folder, env_name, diff_level, seq):
    # The format of each line is 'tx ty tz qx qy qz qw'
    pose_file = os.path.join(root_folder, env_name, diff_level, seq, 'pose_left.txt')
    pose = np.loadtxt(pose_file)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Trajectory of TartanAir %s/%s/%s' % (env_name, diff_level, seq))

    # Data for a three-dimensional line
    pose_x = pose[:, 0]
    pose_y = pose[:, 1]
    pose_z = pose[:, 2]
    ax.plot3D(pose_x, pose_y, pose_z, 'gray')

    plt.savefig('traj/tartanair_'+env_name+'_'+diff_level+'_'+seq+'.png')
    plt.close(fig)


if __name__ == "__main__":
    dataset_root_folder = '/home/cel/DockerFolder/data/tartanair/'
    environment_list = [env_name for env_name in os.listdir(dataset_root_folder) if os.path.isdir(os.path.join(dataset_root_folder, env_name))]
    for env_name in environment_list:
        for diff_level in ['Easy', 'Hard']:
            seq_list = [seq for seq in os.listdir(os.path.join(dataset_root_folder, env_name, diff_level)) if os.path.isdir(os.path.join(dataset_root_folder, env_name, diff_level, seq))]
            for seq in seq_list:
                visualize_trajectory(dataset_root_folder, env_name ,diff_level, seq)