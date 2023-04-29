import torch
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset


class GraspDataset(Dataset):
    def __init__(self, train: bool=True) -> None:
        '''Dataset of successful grasps.  Each data point includes a 64x64
        top-down RGB image of the scene and a grasp pose specified by the gripper
        position in pixel space and rotation (either 0 deg or 90 deg)

        The datasets are already created for you, although you can checkout
        `collect_dataset.py` to see how it was made (this can take a while if you
        dont have a powerful CPU).
        '''
        mode = 'train' if train else 'val'
        self.train = train
        data = np.load(f'{mode}_dataset.npz')
        self.imgs = data['imgs']
        self.actions = data['actions']

    def transform_grasp(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        '''Randomly rotate grasp by 0, 90, 180, or 270 degrees.  The image can be
        rotated using `TF.rotate`, but you will have to do some math to figure out
        how the pixel location and gripper rotation should be changed.

        Arguments
        ---------
        img:
            float tensor ranging from 0 to 1, shape=(3, 64, 64)
        action:
            array containing (px, py, rot_id), where px specifies the row in
            the image (heigh dimension), py specifies the column in the image (width dimension),
            and rot_id is an integer: 0 means 0deg gripper rotation, 1 means 90deg rotation.

        Returns
        -------
        tuple of (img, action) where both have been transformed by random
        rotation in the set (0 deg, 90 deg, 180 deg, 270 deg)

        Note
        ----
        The gripper is symmetric about 180 degree rotations so a 180deg rotation of
        the gripper is equivalent to a 0deg rotation and 270 deg is equivalent to 90 deg.

        Example Action Rotations
        ------------------------
        action = (32, 32, 1)
         - Rot   0 deg : rot_action = (32, 32, 1)
         - Rot  90 deg : rot_action = (31, 32, 0)
         - Rot 180 deg : rot_action = (31, 31, 1)
         - Rot 270 deg : rot_action = (32, 31, 0)

         action = (42, 42, 0)
         - Rot   0 deg : rot_action = (42, 42, 0)
         - Rot  90 deg : rot_action = (21, 42, 1)
         - Rot 180 deg : rot_action = (21, 21, 0)
         - Rot 270 deg : rot_action = (42, 21, 1)

        action = (15, 45, 1)
         - Rot   0 deg : rot_action = (15, 45, 1)
         - Rot  90 deg : rot_action = (18, 15, 0)
         - Rot 180 deg : rot_action = (48, 18, 1)
         - Rot 270 deg : rot_action = (45, 48, 0)
        '''
        ################################
        # Implement this function for Q4
        ################################
        # Randomly choose a rotation angle from the set {0, 90, 180, 270}.
        # angle = int(np.random.choice([0, 90, 180, 270]))
        #
        # # Rotate the image using the chosen angle.
        # rotated_img = TF.rotate(img, angle)
        #
        # # Extract the action components.
        # px, py, rot_id = action
        # H, W = img.shape[1], img.shape[2]
        #
        # # Transform the action components based on the chosen angle.
        # if angle == 0:
        #     # No transformation required.
        #     rot_px = px
        #     rot_py = py
        # elif angle == 90:
        #     # Swap px and py and update the row index.
        #     rot_px = W - px - 1
        #     rot_py = px
        #     # Toggle the rot_id between 0 and 1.
        #     rot_id = 1 - rot_id
        # elif angle == 180:
        #     # Invert both px and py indices.
        #     rot_px = H - px - 1
        #     rot_py = W - py - 1
        # elif angle == 270:
        #     # Swap px and py and update the column index.
        #     rot_px = py
        #     rot_py = H - py - 1
        #     # Toggle the rot_id between 0 and 1.
        #     rot_id = 1 - rot_id
        #
        # # # Translate the real coordinates of the grasp.
        # # x = W - (py + 0.5) * (W / H)
        # # y = (H - px - 0.5) * (W / H)
        #
        # # Create the rotated action array.
        # rotated_action = np.array([rot_px, rot_py, rot_id])
        #
        # return rotated_img, rotated_action
        return img, action

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = self.imgs[idx]
        action = self.actions[idx]

        H, W = img.shape[:2]
        img = TF.to_tensor(img)
        if np.random.rand() < 0.5:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        if self.train:
            img, action = self.transform_grasp(img, action)

        px, py, rot_id = action
        label = np.ravel_multi_index((rot_id, px, py), (2, H, W))

        return img, label

    def __len__(self) -> int:
        '''Number of grasps within dataset'''
        return self.imgs.shape[0]
