# egodex-retargeting

The script retargets EgoDex data to embodiment specific action space.
Currently, we only support XArm + LeapHand hardware setting.

The action space consists of left and right of wrist pose and finger joint angles, which is in total `2 * (6 + 16) = 44` DoF.
Wrist pose is transformed into a camera coordinate. e.g. location w.r.t. AppleVisionPro.

# Installation

```bash
git clone --recurse-submodules git@github.com:yjang43/egodex-retargeting.git

# Or
git clone git@github.com:yjang43/egodex-retargeting.git
git submodule update --init --recursive

```

```bash
cd egodex-retargeting
pip install uv
uv pip install dex-retargeting h5py pytransform3d tqdm torch
```

# Run

```bash
python retarget.py --data_dirpath PATH_TO_EGODEX_DATA_DIR --out_dirpath PATH_TO_OUTPUT_DIR
```

# Action Space
Each output file contains retarged action sequence of an episode.
The naminig of the file is in `{split}/{task_name}/{episode_id}.hdf5` format.

The output file is structured this way,
```python
HDF5 file:
wrist_left_pose (Dataset) shape=(T, 6) dtype=float64
wrist_right_pose (Dataset) shape=(T, 6) dtype=float64
fingers_left_qpos (Dataset) shape=(T, 16) dtype=float64
fingers_right_qpos (Dataset) shape=(T, 16) dtype=float64
```

* wrist_pose: "6-DoF cartesian + axis-angle in camera frame, EE aligned with xArm"
* finger_joint_angle: "16-DoF retargeted with DexPilot; joints 0-15."
