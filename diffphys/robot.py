import numpy as np
import pdb
import torch
import torch.nn as nn
from urdfpy import URDF
from diffphys.urdf_utils import get_joints, robot2parent_idx


class URDFRobot(nn.Module):
    def __init__(self, urdf_path):
        super(URDFRobot, self).__init__()
        self.urdf = URDF.load(urdf_path)
        robot_name = urdf_path.split("/")[-1][:-5]
        self.urdf.robot_name = robot_name
        if robot_name == "human" or robot_name == "quad":
            self.urdf.ball_joint = True  # whether has sperical joints
        else:
            self.urdf.ball_joint = False
        joints = get_joints(self.urdf)  # joints: joint location, for training
        self.urdf.parent_idx = robot2parent_idx(self.urdf)  # visualization only

        # order of operations: scale => rotate => translate
        if robot_name == "a1":
            sim3 = torch.Tensor(
                [0, 0, 0, 0.5, -0.5, -0.5, -0.5, -1.61, -1.61, -1.61]
            )  # center, orient, scale
            self.num_dofs = joints.shape[0]
            rest_angles = torch.zeros(1, joints.shape[0])
            rest_angles[0, 2] = -0.8
            rest_angles[0, 5] = -0.8
            rest_angles[0, 8] = -0.8
            rest_angles[0, 11] = -0.8
        elif robot_name == "laikago":
            sim3 = torch.Tensor(
                [0, 0, 0, 1, 0, 0, 0, -1.61, -1.61, -1.61]
            )  # center, orient, scale
            self.num_dofs = joints.shape[0]
            rest_angles = torch.zeros(1, joints.shape[0])
            rest_angles[0, 2] = -0.8
            rest_angles[0, 5] = -0.8
            rest_angles[0, 8] = -0.8
            rest_angles[0, 11] = -0.8
        elif (
            robot_name == "laikago_toes_zup_joint_order" or robot_name == "laikago_mod"
        ):
            sim3 = torch.Tensor(
                [0, 0, 0, 0.5, -0.5, -0.5, -0.5, -1.61, -1.61, -1.61]
            )  # center, orient, scale
            self.num_dofs = joints.shape[0]
            rest_angles = torch.zeros(1, joints.shape[0])
            rest_angles[0, 2] = -0.8
            rest_angles[0, 5] = -0.8
            rest_angles[0, 8] = -0.8
            rest_angles[0, 11] = -0.8
        elif robot_name == "quad":
            sim3 = torch.Tensor(
                [0, 0.01, -0.04, 0.5, 0.6, 0, 0, -3.1, -3.1, -3.1]
            )  # center, orient, scale
            self.num_dofs = joints.shape[0] * 3
            rest_angles = torch.zeros(1, self.num_dofs)  # ball joints
            self.urdf.kp_links = [
                "link_155_Vorderpfote_R_Y",
                "link_150_Vorderpfote_L_Y",
                "link_170_Pfote2_R_Y",
                "link_165_Pfote2_L_Y",
            ]
            self.urdf.query_links = [
                "link_155_Vorderpfote_R_Y",
                "link_150_Vorderpfote_L_Y",
                "link_170_Pfote2_R_Y",
                "link_165_Pfote2_L_Y",
            ]
        elif robot_name == "human":
            sim3 = torch.Tensor(
                [0, 0, 0, 1, 0, 0, 0, -3.2, -3.2, -3.2]
            )  # center, orient, scale
            self.num_dofs = joints.shape[0] * 3
            rest_angles = torch.zeros(1, self.num_dofs)  # ball joints
            self.urdf.kp_links = [
                "link_24_mixamorig:RightFoot_Y",
                "link_19_mixamorig:LeftFoot_Y",
            ]
            self.urdf.query_links = [
                "link_24_mixamorig:RightFoot_Y",
                "link_19_mixamorig:LeftFoot_Y",
                "link_16_mixamorig:RightHand_Y",
                "link_12_mixamorig:LeftHand_Y",
            ]
        else:
            raise NotImplementedError

        # self.sim3 = sim3# [:8] # 10 => 8
        self.sim3 = sim3[:8]  # 10 => 8
        self.joints = joints
        self.rest_angles = rest_angles
        self.num_bones = len(self.joints) + 1

        # map from body to unique bodies
        unique_body_idx = list(range(len(self.urdf.links)))
        if self.urdf.ball_joint:
            unique_body_idx = unique_body_idx[0:1] + unique_body_idx[3::3]
        self.urdf.unique_body_idx = unique_body_idx

        # assign symmetric index
        if self.urdf.robot_name == "a1" or self.urdf.robot_name == "laikago":
            symm_idx = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        elif self.urdf.robot_name == "quad":
            symm_idx = [
                0,
                1,
                2,
                3,
                8,
                9,
                10,
                11,
                4,
                5,
                6,
                7,
                12,
                13,
                14,
                15,
                16,
                21,
                22,
                23,
                24,
                17,
                18,
                19,
                20,
            ]
        elif self.urdf.robot_name == "human":
            symm_idx = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 15, 16, 17, 12, 13, 14]
        self.urdf.symm_idx = symm_idx
