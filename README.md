# tartanair_place_recognition_tool
This is a repository that helps setting up [TartanAir](https://theairlab.org/tartanair-dataset/) dataset for place recognition task

## TartanAir Dataset
Download data using [tartanair_tools](https://github.com/castacks/tartanair_tools) repository. For place recognition task, only `image_left/` and `depth_left/`, and `pose_left.txt` is necessary. Those data can be downloaded using `python download_training.py --output-dir SAVE_DIR --rgb --depth --only-left` in [tartanair_tools](https://github.com/castacks/tartanair_tools) repository.

## Visualization Tool
To visualize the trajectory of each sequence, change the root folder direction in [visualize_trajectory.py](visualize_trajectory.py) and run `python visualize_trajectory.py`. The visualization figures are saved in [traj/](traj/) folder.

## Generate Point Clouds from RGBD images
For now (May 2022), TartanAir dataset only provides image datas but not LiDAR data. Thus, converting RGBD images to point clouds are necessary to perform place recognition on 3D point clouds.
Note: open3d library is used in this script.
```
python rgbd2pointcloud.py
```

## Geneate Place Recognition Training and Testing Tuplets
For place recognition task, representation learning-based methods needs tuplets for training and testing. To be more specific, for every query point cloud, pair it with near by point clouds (positive) and far away point clouds (negative) to enable network to train with triplet loss or quadruplet loss. 
```
python gen_tuplets.py
```