import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

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
    [0, np.pi, 0], 0, 1, 2, extrinsic=True)
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

LEAP_JOINT_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "11", "12", "13", "14", "15"
]
ALLEGRO_JOINT_NAMES = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
]
JOINT_NAMES = LEAP_JOINT_NAMES


# For Sapien rendering.
FHD = (1920, 1080)
FPS = 30
TIMESTEP = 1.0 / FPS
