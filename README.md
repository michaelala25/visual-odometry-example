# Visual Odometry
This repo contains an example visual odometry system implemented in python, based loosely off work by Christian Forster et al. on "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry".

[Examle output video](https://drive.google.com/file/d/11G_H6GwXcmhNFyoeUJXj8ZadJx0zUQ9a/view?usp=sharing)

**Remark:** This repo was made quickly for a little school project, and is not intended to be anything more than a basic demo of the work we did. Apologies in advance!

The main entry point is the `feature_tracking.py` file, which processes a stereo video from the EuRoC dataset into a `landmarks.csv` file.

The resulting `landmarks.csv` file contains the features identified in the video, tracked over time. Each row of the file contains the following information (in order)
- ID of the camera pose
- Timestamp of the camera pose (aligns with the "timestamp" column in EuRoC's ground truth camera pose csv file - NOT with the video frame timestamps (which are slightly off))
- ID of the tracked feature
- x position of the feature in the left camera
- x position of the feature in the right camera
- y position of the feature (assumed to be identical between cameras, perhaps erroneously)
- x, y, z position of an initial estimate of the landmark's 3D location in space (as determined from the frame in the video with timestamp closest to the camera pose timestamp)

The information in the csv file should be enough to construct a factor graph as in the `StereoVOExample.cpp` in the `GTSAM` library.

You'll need to download one of the [EuRoC datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads) and extract it, as well as update the variables `ROOT_DATA_DIR`, `SUB_DIR_NAME`, as well as potentially `POSE_DIR_NAME`, `LEFT_IMAGE_DIR_NAME`, `RIGHT_IMAGE_DIR_NAME` in `feature_tracking.py.

### Requirements
- `pip install opencv-python`
- `pip install numpy-quaternion`
- `pip install tqdm`
- `pip install pyyaml`


