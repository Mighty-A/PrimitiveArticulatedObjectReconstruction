import cv2
import os
import torch
import torch.nn as nn
import pytorch3d

import math
import torchvision

from networks.vitextractor import ViTExtractor
from networks.graphae import GraphAE

import sys

sys.path.append('../')
from learnable_primitives.equal_distance_sampler_sq import get_sampler
from myutils.tools import compute_rotation_matrix_from_ortho6d


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=False):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer(feat)
        return trans


class RootRot6dPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RootRot6dPredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)

        return out

class RootTransPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RootTransPredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)

        return out


class RootShapePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RootShapePredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)

        return out


class RootSizePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RootSizePredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)

        return out


class LeafShapePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=1, bone_nums=2):
        super(LeafShapePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat)
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class LeafSizePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=1, bone_nums=2):
        super(LeafSizePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList([predictor_block for i in range(self.num_bones)])

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat)
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out

class LeafRotAnglePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bone_nums):
        super(LeafRotAnglePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        # print ('feat.size is', feat.size())
        # [bs, num_bone, in_dim]
        # os._exit(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat)
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            x = torch.tanh(x) * math.pi / 2.

            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out

class RootRotAnglePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bone_nums):
        super(RootRotAnglePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat)
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            x = torch.tanh(x) * math.pi / 2.
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class ViTExtractorCNN(nn.Module):
    def __init__(self, in_channel, out_channel=384):
        super(ViTExtractorCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 4, 2, 1)
        self.group_norm_1 = nn.GroupNorm(64, out_channel)
        self.LRelu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv2d(out_channel, out_channel, 4, 2, 1)
        self.group_norm_2 = nn.GroupNorm(64, out_channel)
        self.LRelu_2 = nn.LeakyReLU(0.2)

        self.conv_3 = nn.Conv2d(out_channel, out_channel, 4, 2, 0)

    def forward(self, feat):
        out = self.conv_1(feat)
        out = self.group_norm_1(out)
        out = self.LRelu_1(out)

        out = self.conv_2(out)
        out = self.group_norm_2(out)
        out = self.LRelu_2(out)

        out = self.conv_3(out)

        return out


class ObjectNetwork_pts(nn.Module):
    def __init__(self, test_mode, model_type, stride, device, vit_f_dim, hidden_dim=256, bone_num=2):
        super(ObjectNetwork_pts, self).__init__()

        print("    Building VIT")
        self.bone_num = bone_num
        self.whole_vitextractor = ViTExtractor(model_type, stride, device=device)
        # self.object_vitextractor = ViTExtractor(model_type, stride, device=device)
        self.vitextractor_fc = nn.Linear(vit_f_dim, vit_f_dim)
        self.vitextractor_cnn = ViTExtractorCNN(vit_f_dim).cuda()

        print("    Building Predictor")
        self.root_rot6d_predictor = RootRot6dPredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=6).cuda()
        self.root_trans_predictor = RootTransPredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=3).cuda()
        self.root_shape_predictor = RootShapePredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=2).cuda()
        self.root_size_predictor = RootSizePredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=3).cuda()
        self.leaf_rotangle_predictor = LeafRotAnglePredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=1, bone_nums=bone_num-1).cuda()
        self.leaf_shape_predictor = LeafShapePredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=2, bone_nums=bone_num-1).cuda()
        self.leaf_size_predictor = LeafSizePredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=3, bone_nums=bone_num-1).cuda()
        self.root_rotangle_predictor = RootRotAnglePredictor(vit_f_dim, hidden_dim=hidden_dim, out_dim=1, bone_nums=bone_num-1).cuda()

    def cal_rotmat_from_angle_around_axis(self, all_joint_axis, all_the_pred_angle):
        batch_size = all_joint_axis.size(0)
        all_rotmat = []
        for i in range(batch_size):
            joint_axis = all_joint_axis[i].unsqueeze(0)
            the_pred_angle = all_the_pred_angle[i]

            axis_2p_sub = torch.sub(joint_axis, torch.zeros(1, 3).cuda())
            axis_len = 0.
            axis_2p_sub = axis_2p_sub.squeeze(0)
            for sub_i in axis_2p_sub:
                axis_len = axis_len + sub_i * sub_i
            axis_len = torch.sqrt(axis_len)
            axis_norm = axis_2p_sub / axis_len

            # R = torch.deg2rad(the_pred_angle)
            R = the_pred_angle
            rot_mat = pytorch3d.transforms.axis_angle_to_matrix(axis_norm * R)
            all_rotmat.append(rot_mat)
        all_rotmat = torch.stack(all_rotmat, dim=0)

        return all_rotmat

    def forward(self, rgb_image):
        batch_size = rgb_image.size(0)
        # extract feature of the whole image

        ################### for DINOv2 ###################
        transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=0.5, std=0.2)])
        rgb_image = transform(rgb_image)
        whole_result = self.whole_vitextractor.forward_features(rgb_image)
        vit_feature_whole = whole_result['x_norm_patchtokens']
        vit_feature_whole = torch.reshape(vit_feature_whole, (batch_size, 16, 16, -1))

        total_vit_feature = vit_feature_whole #  + vit_feature_object

        # # check.
        total_vit_feature = torch.permute(total_vit_feature, (0, 3, 1, 2))
        total_vit_feature = self.vitextractor_cnn(total_vit_feature)
        total_vit_feature = torch.reshape(total_vit_feature, (batch_size, -1))

        pred_root_rot6d = self.root_rot6d_predictor(total_vit_feature)
        pred_root_trans = self.root_trans_predictor(total_vit_feature)
        pred_root_shape = self.root_shape_predictor(total_vit_feature)
        pred_root_size = self.root_size_predictor(total_vit_feature)

        # joint rotation angle 
        pred_root_rot_angle = self.root_rotangle_predictor(total_vit_feature)
        pred_leaf_rot_angle = self.leaf_rotangle_predictor(total_vit_feature)
        
        
        pred_leaf_shape = self.leaf_shape_predictor(total_vit_feature)
        pred_leaf_size = self.leaf_size_predictor(total_vit_feature)

        return (
            pred_root_rot6d,
            pred_root_trans,
            pred_root_shape,
            pred_root_size,
            pred_root_rot_angle,
            pred_leaf_rot_angle,
            pred_leaf_shape,
            pred_leaf_size
        )


class Network_pts(nn.Module):
    def __init__(self,
                 test_mode,
                 model_type,
                 stride,
                 device,
                 vit_f_dim,
                 hidden_dim=256, object_bone_num=2):
        super(Network_pts, self).__init__()

        self.object_network = ObjectNetwork_pts(test_mode, model_type, stride, device, vit_f_dim, hidden_dim=hidden_dim, bone_num=object_bone_num)

    def forward(self,
                rgb_image
                ):

        (
            pred_root_rot6d,
            pred_root_trans,
            pred_root_shape,
            pred_root_size,
            pred_root_rot_angle,
            pred_leaf_rot_angle,
            pred_leaf_shape,
            pred_leaf_size,
        ) = self.object_network(rgb_image)

        pred_dict = {}
        # pack result into a dict
        pred_dict['pred_root_rot6d'] = pred_root_rot6d
        pred_dict["pred_root_trans"] = pred_root_trans
        pred_dict['pred_root_shape'] = pred_root_shape
        pred_dict["pred_root_size"] = pred_root_size
        pred_dict["pred_root_rot_angle"] = pred_root_rot_angle
        pred_dict["pred_leaf_rot_angle"] = pred_leaf_rot_angle
        pred_dict["pred_leaf_shape"] = pred_leaf_shape
        pred_dict['pred_leaf_size'] = pred_leaf_size

        return pred_dict
