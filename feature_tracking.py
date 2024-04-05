from collections import defaultdict
import csv
from pathlib import Path
import os

import cv2
import numpy as np
import quaternion
from tqdm import tqdm
import yaml

# Load stereo images from EuRoC
# at certain intervals run cv2.goodFeaturesToTrack
# every other frame use cv2.calcOpticalFlowPyrLK
# produce feature tracks

# output is a list of landmarks
#   each landmark is a list of values:
#       <pose id> <track id> <x-left> <x-right> <y> <initial estimate x> <initial estimate y> <initial estimate z>
#
#   track id is obtained via feature detection + tracking
#   initial estimate x, y, z are obtained how?
#       - use camera ground truth pose + stereo

ROOT_DATA_DIR = Path("/mnt/c/Users/micha/Downloads/MH_01_easy")
SUB_DIR_NAME = "mav0"
POSE_DIR_NAME = "state_groundtruth_estimate0"
LEFT_IMAGE_DIR_NAME = "cam0"
RIGHT_IMAGE_DIR_NAME = "cam1"


def estimate_3d_from_stereo(left, right, sensor_data, image_size):

    left_intrinsics = sensor_data["left"]["intrinsics"]
    right_intrinsics = sensor_data["right"]["intrinsics"]
    extrinsics = sensor_data["extrinsics"]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_intrinsics["matrix"], left_intrinsics["distortion"],
        right_intrinsics["matrix"], right_intrinsics["distortion"],
        image_size,
        extrinsics["R"], extrinsics["T"]
    )

    left_undistorted_points = cv2.undistortPoints(left, left_intrinsics["matrix"], left_intrinsics["distortion"], R = R1, P = P1)
    right_undistorted_points = cv2.undistortPoints(right, right_intrinsics["matrix"], right_intrinsics["distortion"], R = R2, P = P2)

    points_3d = cv2.triangulatePoints(P1, P2, left_undistorted_points, right_undistorted_points)
    # Convert from homogeneous to 3D, swap dimensions, and remove homogeneous dim
    points_3d /= points_3d[3]
    return points_3d.T[:, :3] 

def cam_space_to_world(points, pose):
    camera_position, camera_rotation = pose
    rotated_points = quaternion.rotate_vectors(camera_rotation, points)
    world_points = rotated_points + camera_position
    return world_points

def cvt_to_pose_data(row):
    # Remark: when the row gets passed in here we drop the first column (the timestamp), so we index from 0
    camera_position = np.array([ # x,y,z
        float(row[0]),  # p_RS_R_x [m]
        float(row[1]),  # p_RS_R_y [m]
        float(row[2])   # p_RS_R_z [m]
    ])
    # Extracting camera's rotation quaternion
    camera_rotation_quaternion = np.quaternion(
        float(row[3]),  # q_RS_w []
        float(row[4]),  # q_RS_x []
        float(row[5]),  # q_RS_y []
        float(row[6])   # q_RS_z []
    )
    return (camera_position, camera_rotation_quaternion)

def get_sensor_data(
    root = ROOT_DATA_DIR,
    sub_dir = SUB_DIR_NAME,
    left_images_dir_name = LEFT_IMAGE_DIR_NAME,
    right_images_dir_name = RIGHT_IMAGE_DIR_NAME
):
    # Here we return a nested dictionary:
    # camera_intrinsics = {
    #   "left" : {
    #       "matrix" : ...,
    #       "distortion" : ...
    #    },
    #    "right" : {...}
    # }

    sensor_data = {}
    
    for direction, sensor_yaml_path in [
        ("left",  root / sub_dir / left_images_dir_name / "sensor.yaml"),
        ("right", root / sub_dir / right_images_dir_name / "sensor.yaml")
    ]:
        sensor_data[direction] = {}

        with open(sensor_yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        fu, fv, cu, cv = data["intrinsics"]
        sensor_data[direction]["intrinsics"] = {
            "matrix" : np.array([
                [fu, 0, cu],
                [0, fv, cv],
                [0, 0, 1]
            ]),
            "distortion" : np.array(data["distortion_coefficients"])
        }

        extrinsic_matrix = np.array(data["T_BS"]["data"]).reshape(4, 4)
        sensor_data[direction]["extrinsics"] = {
            "rotation" : extrinsic_matrix[:3, :3],
            "translation" : extrinsic_matrix[0:3, 3]
        }

    # transformations between left and right cameras
    R = np.matmul(
        sensor_data["right"]["extrinsics"]["rotation"], 
        sensor_data["left"]["extrinsics"]["rotation"].T
    )
    T = sensor_data["right"]["extrinsics"]["translation"] - np.dot(R, sensor_data["left"]["extrinsics"]["translation"])
    sensor_data["extrinsics"] = {
        "R" : R,
        "T" : T
    }

    return sensor_data

def infer_camera_extrinsics(intrinsics):
    # Here we infer the 
    rot_R = intrinsics["right"]["matrix"][:3, :3]
    rot_L = intrinsics["left"]["matrix"][:3, :3]

    trans_R = intrinsics["right"]["matrix"][:3, 3]
    trans_L = intrinsics["left"]["matrix"][:3, 3]

    R = np.matmul(rot_R, rot_L.T)
    T = trans_R - np.dot(R, trans_L)
    return {"R" : R, "T" : T}

def find_nearest_pose(image_timestamp, pose_timestamps, poses):
    nearest_index = np.abs(pose_timestamps - image_timestamp).argmin()
    return poses[nearest_index], pose_timestamps[nearest_index]

def create_frame_data_iterator(
    root = ROOT_DATA_DIR,
    sub_dir = SUB_DIR_NAME,
    pose_dir_name = POSE_DIR_NAME,
    left_images_dir_name = LEFT_IMAGE_DIR_NAME,
    right_images_dir_name = RIGHT_IMAGE_DIR_NAME
):
    # Need to produce something which yields tuples (left frame, right frame, cam pose)
    # For our purposes, the frames yielded can be grayscale


    # First we load the camera pose data
    cam_data_file_path = root / sub_dir / pose_dir_name / "data.csv"
    pose_data = []
    with open(cam_data_file_path, "r", newline="") as cam_data_file:
        cam_data_reader = csv.reader(cam_data_file)
        next(cam_data_reader)  # skip header
        pose_data = [(int(row[0]), row[1:]) for row in cam_data_reader]

    pose_timestamps = np.array([p[0] for p in pose_data])
    poses = [p[1] for p in pose_data]

    # The timestamps in the pose data csv don't align with the filenames in the camera data
    # directory, so we need to interpolate

    left_images_dir = root / sub_dir / left_images_dir_name / "data"
    right_images_dir = root / sub_dir / right_images_dir_name / "data"

    left_image_files = sorted(os.listdir(left_images_dir))
    right_image_files = sorted(os.listdir(right_images_dir))

    for left_image_file, right_image_file in zip(left_image_files, right_image_files):
        left_image_timestamp = int(left_image_file.split('.')[0])
        # right_image_timestamp = int(right_image_file.split('.')[0])

        # Find nearest pose
        nearest_pose, pose_timestamp = find_nearest_pose(left_image_timestamp, pose_timestamps, poses)

        # Now we read the left/right frames
        left = cv2.imread(str(left_images_dir / left_image_file))
        right = cv2.imread(str(right_images_dir / right_image_file))

        # Convert to grayscale
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Yield
        yield left, right, cvt_to_pose_data(nearest_pose), pose_timestamp

    cam_data_file.close()

# Remark: this is a tuple because it's the default value of a function argument below, and so needs to be (or rather should be) immutable
DEFAULT_SHI_TOMASI_PARAMS = (
    ("maxCorners", 30),
    ("qualityLevel", 0.3),
    ("minDistance", 7),
    ("blockSize", 7)
)

DEFAULT_LUCAS_KANADE_PARAMS = (
    ("winSize", (15, 15)),
    ("maxLevel", 2),
    ("criteria", (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    ))
)

def filter_duplicates(new_features, current_points, threshold = 10):
    new_features = new_features.reshape(-1, 2)
    current_points = current_points.reshape(-1, 2)

    # Remove any new feature points that fall within the threshold of any of the current points
    filtered_features = []
    for new_point in new_features:
        if any(np.linalg.norm(point - new_point) < threshold for point in current_points):
            continue
        filtered_features.append(new_point)

    if not filtered_features:
        return np.array(filtered_features)
    filtered_features = np.array(filtered_features)[:, None, :] # reinsert the middle dim
    return filtered_features


def track_features(
    frame_data_iterator, 
    sensor_data,
    num_frames,
    keyframe_frequency = 10, 
    shi_tomasi_params = DEFAULT_SHI_TOMASI_PARAMS, 
    lucas_kanade_params = DEFAULT_LUCAS_KANADE_PARAMS,
    duplicate_threshold = 10,
    min_tracked_points = 10,
    max_tracked_points = 100,
    visualize = False,
    video_file = "output.mp4"
):
    if isinstance(shi_tomasi_params, tuple):
        shi_tomasi_params = dict(shi_tomasi_params)
    if isinstance(lucas_kanade_params, tuple):
        lucas_kanade_params = dict(lucas_kanade_params)

    prev_left = None
    landmarks = defaultdict(list) # Dictionary to hold frame index to landmarks mapping
    current_tracks = [] # List of the currently tracked points
    current_points = None

    if visualize:
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_file, fourcc, 20.0, (752, 480)) # Hard coded for now

    for i, (left, right, pose, pose_timestamp) in enumerate(tqdm(frame_data_iterator, desc = "Tracking Features", total = num_frames)):

        # here's how this works:
        #
        #   - At each frame we have a list of the currently tracked ids, which map 1:1 with the
        #     currently tracked points. That is, current_points[i] is the current point of the track with id
        #     current_tracks[i], for all i (len(current_points) == len(current_tracks))
        #
        #   - First, we do optical flow matching between the prev_left frame and the current left frame.
        #     Points which aren't matched between the prev and current frame are discarded (their track ids)
        #     are removed from the list `current_tracks`.
        #
        #   - Next, we do optical flow matching between left and right frame to obtain stereo estimates of the
        #     point locations in 3D space. If a point isn't matched, we don't dsicard the track, we just don't
        #     write any new point to the track in the `landmarks` dict.
        #
        #   - At keyframe intervals or if the number of tracked points falls below a certain threshold
        #     we rerun feature detection and filter duplicates, potentially obtaining new tracks

        if i == 0:
            # Detect initial features and initialize tracking
            current_points = cv2.goodFeaturesToTrack(left, **shi_tomasi_params)
            current_tracks = list(range(len(current_points)))
            prev_left = left.copy()
            continue

        new_points, temporally_matched, _ = cv2.calcOpticalFlowPyrLK(
            prev_left, left, current_points, None, **lucas_kanade_params
        )
        current_points = new_points[temporally_matched[:, 0] == 1]
        current_tracks = [current_tracks[i] for i in range(len(temporally_matched)) if temporally_matched[i]]

        if (i > 0 and i % keyframe_frequency == 0) or len(current_tracks) < min_tracked_points:
            new_features = cv2.goodFeaturesToTrack(left, **shi_tomasi_params)
            unique_new_features = filter_duplicates(new_features, current_points, threshold = duplicate_threshold)
            
            if unique_new_features.size:
                # It's also possible that we obtain _too many_ new features, so we need to cull
                culled_new_features = unique_new_features[:max_tracked_points - len(current_tracks) - len(unique_new_features)]
                
                # Extend the tracks
                if culled_new_features.size:
                    current_points = np.append(current_points, culled_new_features, axis = 0)
                    start = max(landmarks.keys()) + 1
                    stop = start + len(culled_new_features)
                    current_tracks.extend(range(start, stop))
                    assert len(current_points) == len(current_tracks)

        right_points, stereo_matched, _ = cv2.calcOpticalFlowPyrLK(
            left, right, current_points, None, **lucas_kanade_params
        )
        stereo_matched_left_points = current_points[stereo_matched[:, 0] == 1]
        stereo_matched_right_points = right_points[stereo_matched[:, 0] == 1]
        points_3d = estimate_3d_from_stereo(
            stereo_matched_left_points,
            stereo_matched_right_points,
            sensor_data,
            left.shape
        )
        # Convert from cam space to world space
        world_points = cam_space_to_world(points_3d, pose)

        for track_id, (x1, y), (x2, y), (ix, iy, iz) in zip(
            current_tracks,
            stereo_matched_left_points.reshape(-1, 2),
            stereo_matched_right_points.reshape(-1, 2),
            world_points
        ):
            landmarks[track_id].append(
                (i, pose_timestamp, x1, x2, y, ix, iy, iz)
            )

        prev_left = left.copy()

        # Visualize
        if visualize and i > 1000 and i < 2000:
            frame = cv2.cvtColor(left.copy(), cv2.COLOR_GRAY2BGR)
            for track_id, (x, y) in zip(current_tracks, stereo_matched_left_points.reshape(-1, 2)):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(frame, str(track_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            video_writer.write(frame)

    video_writer.release()
    return landmarks
        
def plot_3d(points_lists, filename):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for points in points_lists:
        x, y, z = zip(*points)
        ax.plot(x, y, z)
    
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.savefig(filename)

if __name__ == "__main__":
    
    sensor_data = get_sensor_data()
    frame_data_iterator = create_frame_data_iterator()

    landmarks = track_features(
        frame_data_iterator,
        sensor_data,
        3682, # hardcoded for now
        visualize = True
    )

    # Visualize some 3D lines
    keys = list(sorted(landmarks.keys()))[10:20]
    points_lists = [
        [(ix, iy, iz) for *_, ix, iy, iz in landmarks[id]][:10]
        for id in keys
    ]
    plot_3d(points_lists, filename = "3d.png")

    # Now write the landmarks to a file

    # Remark: at this stage in the code we have a bit of a representation problem before passing it off to the next layer (which produces the
    # factor graph) - the problem is that the track ids and the pose ids are not disjoint sets. In other words, we can have a track id and a
    # pose id which are the same, and this is a problem because the factor graph will be unable to distinguish the two.

    # To fix this we simply offset the pose ids
    offset = max(landmarks.keys()) + 1

    with open("landmarks.csv", "w", newline="") as csvfile:
        landmark_writer = csv.DictWriter(csvfile, delimiter=",", fieldnames = [
            "pose_idx", "pose_timestamp", "track_id", "x1", "x2", "y", "ix", "iy", "iz"
        ])
        landmark_writer.writeheader()

        for track_id in landmarks:
            for pose_idx, pose_timestamp, x1, x2, y, ix, iy, iz in landmarks[track_id]:
                landmark_writer.writerow({
                    "pose_idx" : pose_idx + offset,
                    "pose_timestamp" : pose_timestamp,
                    "track_id" : track_id,
                    "x1" : x1,
                    "x2" : x2,
                    "y" : y,
                    "ix" : ix,
                    "iy" : iy,
                    "iz" : iz
                })
