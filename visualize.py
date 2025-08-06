"""Render retargeting result in Sapien simulator."""

from pathlib import Path
from argparse import ArgumentParser
import time

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from sapien import Pose
from sapien.utils import Viewer
import h5py
import cv2
import numpy as np
import sapien
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from const import *



def render_file(
    filepath: str,
    data_dirpath: str,
    retargeting_dirpath: str,
    headless: bool = False,
    video_filepath: str = "",
    robot_name: str = "leap",
) -> None:

    # Load trajectory.
    file_name = Path(filepath).name
    task_name = Path(filepath).parent.name
    split = Path(filepath).parent.parent.name

    retargeting_filepath = str(Path(retargeting_dirpath) / split / task_name / file_name)
    with h5py.File(retargeting_filepath, "r") as rfile:
        left_qpos = rfile["fingers_left_qpos"][:]
        right_qpos = rfile["fingers_right_qpos"][:]
        left_pose = rfile["wrist_left_pose"][:]
        right_pose = rfile["wrist_right_pose"][:]
        T = left_qpos.shape[0]
    
    # Load camera parameters.
    filepath = str(Path(data_dirpath) / split / task_name / file_name)
    with h5py.File(filepath, "r") as file:
        extrinsic = file["transforms/camera"][:]
        intrinsic = file["camera/intrinsic"][:]

    # Setup sapien environment.
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # Load robot hands.
    robot_dir = Path(__file__).absolute().parent / "dex-urdf" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    left_config_path = get_default_config_path(
        RobotName[robot_name],
        RetargetingType.dexpilot,
        HandType.left
    )
    right_config_path = get_default_config_path(
        RobotName[robot_name],
        RetargetingType.dexpilot,
        HandType.right
    )
    left_urdf_path = RetargetingConfig.load_from_file(left_config_path).urdf_path
    right_urdf_path = RetargetingConfig.load_from_file(right_config_path).urdf_path

    loader = scene.create_urdf_loader()
    left_robot = loader.load(left_urdf_path)
    right_robot = loader.load(right_urdf_path)

    # Load camera.
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    fovy = 2 * np.arctan(cy / fy)
    camera = scene.add_camera(
        name="camera", width=FHD[0], height=FHD[1], fovy=fovy, near=0.1, far=10
    )

    # Setup onscreen viewer if not headless
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.paused = True

    # Setup video recorder
    if video_filepath:
        writer = cv2.VideoWriter(
            video_filepath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            FHD
        )

    # Match the order of joints.
    sapien_left_joint_names = [
        joint.get_name() for joint in left_robot.get_active_joints()
    ]
    sapien_right_joint_names = [
        joint.get_name() for joint in right_robot.get_active_joints()
    ]
    left_retargeting_to_sapien = [
        JOINT_NAMES.index(name) for name in sapien_left_joint_names
    ]
    right_retargeting_to_sapien = [
        JOINT_NAMES.index(name) for name in sapien_right_joint_names
    ]

    for i in range(T):
        # Set camera pose.
        # NOTE: Sapien camera follows different convention.
        cam2world = extrinsic[i]
        sapcam2world = cam2world @ SAPCAM2CAM
        camera.set_pose(Pose(sapcam2world))

        # Set joint qpos.
        left_robot.set_qpos(left_qpos[i][left_retargeting_to_sapien])
        right_robot.set_qpos(right_qpos[i][right_retargeting_to_sapien])

        # Set wrist pose.
        # Wrist pose of XArm is saved in camera coordinate system.
        # We want the wrist pose of Sapien in world coordinate system.
        p, aa = left_pose[i, :3], left_pose[i, 3:]
        q = pr.quaternion_from_compact_axis_angle(aa)
        xarm2cam = pt.transform_from_pq([*p, *q])
        opt2world = cam2world @ xarm2cam @ OPT2XARM
        left_robot.set_pose(Pose(opt2world))

        p, aa = right_pose[i, :3], right_pose[i, 3:]
        q = pr.quaternion_from_compact_axis_angle(aa)
        xarm2cam = pt.transform_from_pq([*p, *q])
        opt2world = cam2world @ xarm2cam @ OPT2XARM
        right_robot.set_pose(Pose(opt2world))

        scene.update_render()

        if not headless:
            t0 = time.perf_counter()
            viewer.render()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, TIMESTEP - elapsed))

        if video_filepath:
            camera = scene.get_cameras()[0]
            camera.take_picture()
            rgb = camera.get_picture_cuda("Color").torch()[..., :3].cpu().numpy()
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if video_filepath:
        writer.release()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dirpath", type=str)
    parser.add_argument("--retargeting_dirpath", type=str)

    parser.add_argument("--filepath", type=str, default="test/add_remove_lid/0.hdf5")
    parser.add_argument("--robot_name", type=str, default="leap")
    parser.add_argument("--out_filepath", type=str, default="asset/retargeted.mp4")

    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)

    args = parser.parse_args()

    assert args.robot_name == "leap", "Currently, only LEAP Hand is supported."

    # data_dirpath = "/home/yjang43/workspace/playground/data/EgoDex"
    # retargeting_dirpath  = "/home/yjang43/workspace/playground/data/EgoDexRetargeted"

    render_file(
        filepath=args.filepath,
        data_dirpath=args.data_dirpath,
        retargeting_dirpath=args.retargeting_dirpath,
        headless=args.headless,
        video_filepath=args.out_filepath,
        robot_name=args.robot_name
    )
