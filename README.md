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

# Download EgoDex
From [EgoDex paper A.3](https://arxiv.org/pdf/2505.11709)
>The training set is further divided into 5 zip files for portability,
each with a size of around 300 GB. The data is publicly available for download at the following URL: https://ml-site.cdn-apple.com/datasets/egodex/[filename]
.zip, where [filename] is one of part1, part2, part3, part4, part5, test,
extra (for example, https://ml-site.cdn-apple.com/datasets/egodex/test.
zip). The 5 parts belong to the training set, whereas test.zip is the test set and extra.zip is
the additional data. 

# Run

```bash
python retarget.py --data_dirpath PATH_TO_EGO_DEX_DATA_DIR --out_dirpath PATH_TO_OUTPUT_DIR
```

# Action Space
Each output file contains retarged action sequence of an episode.
The naminig of the file is in `{split}/{task_name}/{episode_id}.hdf5` format.

The output file is structured this way,
```bash
HDF5 file:
wrist_left_pose (Dataset) shape=(T, 6) dtype=float64
wrist_right_pose (Dataset) shape=(T, 6) dtype=float64
fingers_left_qpos (Dataset) shape=(T, 16) dtype=float64
fingers_right_qpos (Dataset) shape=(T, 16) dtype=float64
```

* wrist_pose: "6-DoF cartesian + axis-angle in camera frame, EE aligned with xArm"
* finger_joint_angle: "16-DoF retargeted with DexPilot; joints 0-15."


# Visualize
To check the quality of the retargeted result, one can visualize with Sapien simulator.
To visualize, it is required to install sapien simulator.
Check out [sapien installation guide](https://sapien-sim.github.io/docs/user_guide/getting_started/installation.html).

```bash
pip install sapien

python visualize.py \
    --filepath PATH_TO_FILE_TO_VISUALIZE    # test/add_remove_lid/0.hdf5 \
    --data_dirpath PATH_TO_EGO_DEX_DATA_DIR \
    --retargeting_dirpath PATH_TO_RETARGETING_DIR \
    --out_filepath PATH_TO_OUTPUT_VIDEO
```

<video src="asset/sample.mp4" controls autoplay loop muted playsinline width="640"></video>

<video src="asset/retargeted.mp4" controls autoplay loop muted playsinline width="640"></video>