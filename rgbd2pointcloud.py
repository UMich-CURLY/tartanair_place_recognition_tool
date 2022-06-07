import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

def rgbd2pointcloud(root_folder, env_name ,diff_level, seq):
    color_folder = os.path.join(root_folder, env_name, diff_level, seq, 'image_left')
    depth_folder = os.path.join(root_folder, env_name, diff_level, seq, 'depth_left')
    print('color_folder', color_folder)
    print('depth_folder', depth_folder)
    num_frames = len([file for file in os.listdir(color_folder) if os.path.isfile(os.path.join(color_folder, file))])
    print('num_frames', num_frames)
    for i in range(num_frames):
        color_raw = o3d.io.read_image(os.path.join(color_folder, str(i).zfill(6)+'_left.png'))
        depth_npy = np.load(os.path.join(depth_folder, str(i).zfill(6)+'_left_depth.npy'))
        # print('depth_npy', depth_npy)
        depth_raw = o3d.geometry.Image(depth_npy.astype(np.uint8))
        # depth_raw = o3d.io.read_image(os.path.join(depth_folder, str(i).zfill(6)+'_left_depth.npy'))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        # print(rgbd_image)
        # save point cloud

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd], zoom=0.5)

if __name__ == "__main__":
    dataset_root_folder = '/home/cel/DockerFolder/data/tartanair/'
    environment_list = [env_name for env_name in os.listdir(dataset_root_folder) if os.path.isdir(os.path.join(dataset_root_folder, env_name))]
    for env_name in environment_list:
        for diff_level in ['Easy', 'Hard']:
            seq_list = [seq for seq in os.listdir(os.path.join(dataset_root_folder, env_name, diff_level)) if os.path.isdir(os.path.join(dataset_root_folder, env_name, diff_level, seq))]
            for seq in seq_list:
                rgbd2pointcloud(dataset_root_folder, env_name ,diff_level, seq)