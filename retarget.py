
from argparse import ArgumentParser
from pathlib import Path
import glob
import os
import multiprocessing as mp
from functools import partial

from tqdm import tqdm
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
import h5py
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import numpy as np

# Transformation Matrices #
# wrist: cam - world - avp - mano - opt - xarm
# fingers: avp - mano - opt
# camera: cam - sapcam

SAPCAM2CAM = np.eye(4)
SAPCAM2CAM[:3, :3] = pr.matrix_from_euler(
    [np.pi/2, -np.pi/2, 0], 0, 1, 2, extrinsic=True)
CAM2SAPCAM = pt.invert_transform(SAPCAM2CAM)

# MANO representation follows,
# https://github.com/dexsuite/dex-retargeting/blob/3f56141bc8bd2760d5e452e382937269554ebb21/example/vector_retargeting/single_hand_detector.py#L130C9-L130C40
# middle to wrist: x ; middle to index: z ; normal: y
MANO_RIGHT2AVP_RIGHT = np.eye(4)
AVP_RIGHT2MANO_RIGHT = np.eye(4)
MANO_LEFT2AVP_LEFT = np.eye(4)
MANO_LEFT2AVP_LEFT[:3, :3] = pr.matrix_from_euler(
    [np.pi, 0, 0], 0, 1, 2, extrinsic=True)
AVP_LEFT2MANO_LEFT = pt.invert_transform(MANO_LEFT2AVP_LEFT)

# Transformation required for motion retargeting algorithm.
# https://github.com/dexsuite/dex-retargeting/issues/13#issuecomment-2133886267
OPT2MANO_RIGHT = np.eye(4)
OPT2MANO_RIGHT[:3, :3] = pr.matrix_from_euler(
    [np.pi/2, 0, -np.pi/2], 0, 1, 2, extrinsic=True)
OPT2MANO_LEFT = np.eye(4)
OPT2MANO_LEFT[:3, :3] = pr.matrix_from_euler(
    [-np.pi/2, 0, np.pi/2], 0, 1, 2, extrinsic=True)

# Embodiment specific transformation. Currently only for XArm.
XARM2OPT = np.eye(4)
XARM2OPT[:3, :3] = pr.matrix_from_euler(
    [np.pi, -np.pi/2, 0], 0, 1, 2, extrinsic=True)
OPT2XARM = pt.invert_transform(XARM2OPT)


def get_avp_joint_names(side): return [
    f"{side}Hand", f"{side}ThumbKnuckle", f"{side}ThumbIntermediateBase", f"{side}ThumbIntermediateTip", f"{side}ThumbTip",
    f"{side}IndexFingerKnuckle", f"{side}IndexFingerIntermediateBase", f"{side}IndexFingerIntermediateTip", f"{side}IndexFingerTip",
    f"{side}MiddleFingerKnuckle", f"{side}MiddleFingerIntermediateBase", f"{side}MiddleFingerIntermediateTip", f"{side}MiddleFingerTip",
    f"{side}RingFingerKnuckle", f"{side}RingFingerIntermediateBase", f"{side}RingFingerIntermediateTip", f"{side}RingFingerTip",
    f"{side}LittleFingerKnuckle", f"{side}LittleFingerIntermediateBase", f"{side}LittleFingerIntermediateTip", f"{side}LittleFingerTip",
]


def get_keypoint_3d(timestep, side="right"):
    keypoint_3d = np.stack([timestep[joint_name][:3, 3] for joint_name in get_avp_joint_names(side)], axis=0)
    return keypoint_3d


def get_wrist_rotation(timestep, side="right"):
    joint_name = f"{side}Hand"
    wrist_rot = timestep[joint_name][:3, :3]
    return wrist_rot


def retarget(retargeting, timestep, side="right"):
    keypoint_3d = get_keypoint_3d(timestep, side)
    wrist_rot = get_wrist_rotation(timestep, side)

    if side == "right":
        mano2avp = MANO_RIGHT2AVP_RIGHT
        opt2mano = OPT2MANO_RIGHT
    elif side == "left":
        mano2avp = MANO_LEFT2AVP_LEFT
        opt2mano = OPT2MANO_LEFT

    # Transform Wrist Pose #
    wrist_pose = np.eye(4)
    wrist_pose[:3, :3] = wrist_rot
    wrist_pose[:3, 3] = keypoint_3d[0, :]

    world2cam = pt.invert_transform(timestep["camera"])
    # NOTE: Adjustment to XArm orientation.
    pose = world2cam @ wrist_pose @ mano2avp @ opt2mano @ XARM2OPT
    rot = pr.compact_axis_angle_from_matrix(pose[:3, :3])
    trans = pose[:3, 3]
    pose = np.concatenate([trans, rot], axis=0)  # cartesian + axis angle

    # Motion Retarget Joint Positions #
    rel_keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]
    joint_pos = rel_keypoint_3d @ wrist_rot @ mano2avp[:3, :3] @ opt2mano[:3, :3]

    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    assert retargeting_type == "DEXPILOT"   # Following BunnyVisionPro Setting.

    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    ref_value = (
        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
    )
    qpos = retargeting.retarget(ref_value)

    # NOTE: Convert back to leaphand joint order 0-15.
    joint_order = [int(j) for j in retargeting.optimizer.robot.dof_joint_names]
    qpos = qpos[joint_order]

    return pose, qpos


def load_hdf5_to_dict(hdf5, index, keys_to_ignore=[]):
    data_dict = {}

    for key in hdf5.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(
                curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


def worker_init(robot_name: str = "leap"):
    global left_retargeting, right_retargeting

    robot_dir = Path(__file__).absolute().parent / "dex-urdf" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    left_config_path = get_default_config_path(
        RobotName[robot_name], RetargetingType.dexpilot, HandType.left)
    right_config_path = get_default_config_path(
        RobotName[robot_name], RetargetingType.dexpilot, HandType.right)
    left_retargeting = RetargetingConfig.load_from_file(
        left_config_path).build()
    right_retargeting = RetargetingConfig.load_from_file(
        right_config_path).build()


def process_file(
    filepath: str,
    data_dirpath: str,
    out_dirpath: str
) -> None:
    file_name = Path(filepath).name
    task_name = Path(filepath).parent.name
    split = Path(filepath).parent.parent.name

    with h5py.File(filepath, "r") as hdf5_file:
        data = hdf5_file["transforms"]
        T = data[next(iter(data.keys()))].shape[0]

        wl_pose, wr_pose = [], []
        fl_qpos, fr_qpos = [], []

        for i in range(T):
            timestep = load_hdf5_to_dict(data, i)
            wl_p, fl_q = retarget(left_retargeting,  timestep, side="left")
            wr_p, fr_q = retarget(right_retargeting, timestep, side="right")

            wl_pose.append(wl_p)
            wr_pose.append(wr_p)
            fl_qpos.append(fl_q)
            fr_qpos.append(fr_q)

        wl_pose = np.stack(wl_pose, axis=0)
        wr_pose = np.stack(wr_pose, axis=0)
        fl_qpos = np.stack(fl_qpos, axis=0)
        fr_qpos = np.stack(fr_qpos, axis=0)

    out_dir = Path(out_dirpath) / split / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir / file_name, "w") as hdf5_out:
        hdf5_out.create_dataset("wrist_left_pose", data=wl_pose)
        hdf5_out.create_dataset("wrist_right_pose", data=wr_pose)
        hdf5_out.create_dataset("fingers_left_qpos", data=fl_qpos)
        hdf5_out.create_dataset("fingers_right_qpos", data=fr_qpos)

        hdf5_out.attrs["wrist_desc"] = \
            "6-DoF cartesian + axis-angle in camera frame, EE aligned with xArm"
        hdf5_out.attrs["finger_desc"] = \
            "16-DoF retargeted with DexPilot; joints 0-15."



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dirpath", type=str)
    parser.add_argument("--out_dirpath", type=str)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    # data_dirpath = "/home/yjang43/workspace/playground/data/EgoDex"
    # out_dirpath  = "/home/yjang43/workspace/playground/data/EgoDexRetargeted"

    filepaths = glob.glob(str(Path(args.data_dirpath) / "**" / "*.hdf5"), recursive=True)

    # n_workers = mp.cpu_count()
    n_workers = args.num_workers

    with mp.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=("leap",)
    ) as pool:

        job = partial(
            process_file,
            data_dirpath=args.data_dirpath,
            out_dirpath=args.out_dirpath
        )

        for _ in tqdm(pool.imap_unordered(job, filepaths), total=len(filepaths)):
            pass       # result is None; loop drives tqdm
