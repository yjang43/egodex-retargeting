#!/usr/bin/env python

from argparse import ArgumentParser
from typing import Tuple
from pathlib import Path
import glob
import multiprocessing as mp
import os

import cv2
import h5py
import numpy as np

from pytransform3d.rotations import quaternion_from_compact_axis_angle
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

from const import FPS, EGODEX_FEATURES


def load_video(video_path: str, target_size=(640, 360)) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == FPS

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)


def to_state_sequence(retarget_data: dict) -> np.ndarray:
    left_xyz = retarget_data["wrist_left_pose"][:, :3]
    left_caa = retarget_data["wrist_left_pose"][:, 3:6]
    left_quat = np.array([quaternion_from_compact_axis_angle(a) for a in left_caa])
    left_joints = retarget_data["fingers_left_qpos"]

    right_xyz = retarget_data["wrist_right_pose"][:, :3]
    right_caa = retarget_data["wrist_right_pose"][:, 3:6]
    right_quat = np.array([quaternion_from_compact_axis_angle(a) for a in right_caa])
    right_joints = retarget_data["fingers_right_qpos"]

    state_seq = np.concatenate([
        left_xyz, left_quat, left_joints,
        right_xyz, right_quat, right_joints
    ], axis=1)

    return state_seq.astype(np.float32)


def process_episode(
    lerobot_dataset: LeRobotDataset,
    episode_info: Tuple[str, str],
    data_dirpath: str,
    retarget_dirpath: str,
    split: str,
    fps: int,
    resolution: tuple,
) -> None:
    task, eps_i = episode_info

    video_path = Path(data_dirpath) / split / task / f"{eps_i}.mp4"
    video = load_video(str(video_path), target_size=resolution)

    retarget_path = Path(retarget_dirpath) / split / task / f"{eps_i}.hdf5"
    with h5py.File(str(retarget_path), "r") as f:
        retarget_data = {
            "wrist_left_pose": f["wrist_left_pose"][:],
            "fingers_left_qpos": f["fingers_left_qpos"][:],
            "wrist_right_pose": f["wrist_right_pose"][:],
            "fingers_right_qpos": f["fingers_right_qpos"][:],
        }
    state_or_action_seq = to_state_sequence(retarget_data)

    num_frames = len(video)
    assert num_frames == retarget_data["wrist_left_pose"].shape[0]

    decimation_ratio = FPS / fps

    indices = np.linspace(
        start=0,
        stop=num_frames - 1,
        num=round(num_frames / decimation_ratio),
        endpoint=True
    )
    indices = np.round(indices).astype(np.int32)

    video = video[indices[:-1]]
    action_seq = state_or_action_seq[indices[1:]]
    state_seq = state_or_action_seq[indices[:-1]]

    for image, state, action in zip(video, state_seq, action_seq):
        frame = {
            "observation.image": image,
            "observation.state": state,
            "action": action,
            "task": task,
        }

        lerobot_dataset.add_frame(frame)

    lerobot_dataset.save_episode()


def process_episodes(
    worker_index: int,
    assigned_episodes: list,
    data_dirpath: str,
    retarget_dirpath: str,
    split: str,
    fps: int,
    resolution: tuple,
    robot_name: str,
) -> None:
    repo_id = f"egodex-{split}-{worker_index}"
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_name,
        fps=fps,
        features=EGODEX_FEATURES[robot_name],
    )

    for episode_info in assigned_episodes:
        process_episode(
            lerobot_dataset,
            episode_info,
            data_dirpath,
            retarget_dirpath,
            split,
            fps,
            resolution,
        )

    lerobot_dataset.finalize()


def main():
    # suppress SVT[INFO] log
    os.environ["SVT_LOG"] = "1"

    parser = ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, required=True)
    parser.add_argument("--retarget_dirpath", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--repo_id", type=str, default="egodex")
    parser.add_argument("--splits", type=str, default="test")       # Later, "train,test" or "part1,part2,test"
    parser.add_argument("--robot_name", type=str, default="leap")
    parser.add_argument("--resolution", type=str, default="640,360")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    assert args.robot_name == "leap", "Currently, only LEAP Hand is supported."

    res = tuple(map(int, args.resolution.split(",")))

    filepaths = glob.glob(str(Path(args.data_dirpath) / "**" / "*.mp4"), recursive=True)

    episodes = {s: [] for s in args.splits.split(",")}
    for filepath in filepaths:
        filepath = Path(filepath)
        eps_i = filepath.stem
        task = filepath.parent.name
        split = filepath.parent.parent.name

        if split in episodes:
            episodes[split].append((task, eps_i))

    num_workers = args.num_workers

    print(
        "==============================================================\n"
        f"Porting EgoDex({args.robot_name}) to LeRobot format dataset...\n"
        "==============================================================\n"
    )

    for split in episodes:
        split_episodes = episodes[split]
        shards = [split_episodes[i::num_workers] for i in range(num_workers)]
        print(f"{num_workers} workers [{', '.join([str(len(s)) for s in shards])}] working together to process {len(split_episodes)} {split} episodes...")

        with mp.Pool(processes=num_workers) as pool:
            worker_args = [
                (i, shards[i], args.data_dirpath, args.retarget_dirpath,
                 split, args.fps, res, args.robot_name)
                for i in range(num_workers)
            ]
            pool.starmap(process_episodes, worker_args)


        repo_id = f"{args.repo_id}-{split}"
        shard_repo_ids = [f"{args.repo_id}-{split}-{i}" for i in range(num_workers)]
        print(f"Aggregating [{', '.join(shard_repo_ids)}] into {repo_id}")
        aggregate_datasets(shard_repo_ids, repo_id)



if __name__ == "__main__":
    main()
