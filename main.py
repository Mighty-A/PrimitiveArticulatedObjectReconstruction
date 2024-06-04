'''
The main code for training the model
'''
import os
import argparse
import numpy as np
import scipy.io
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from tensorboardX import SummaryWriter
import yaml
import json
import torch.utils.data
import string
import random
import datasets
from pytorch3d.io import load_ply
from pytorch3d import renderer
from plyfile import PlyData
from matplotlib import pyplot as plt
import pytorch3d.transforms.rotation_conversions as rc

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from renderer_nvdiff import Nvdiffrast, render
from networks.baseline_network import Network_pts
from myutils.graphAE_param import Parameters

from myutils.tools import compute_rotation_matrix_from_ortho6d
from visualization_utils import points_on_sq_surface_torch

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

file_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Determine if we run the code for training or testing. Chosen from [train, test]')
parser.add_argument('--log_dir', type=str, default=os.path.join(file_dir, "logs"))
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--output_directory', type=str, default='../../NewPrimReg_outputs_iccv/baseline/output_dir')
parser.add_argument('--output_directory', type=str, default=os.path.join(file_dir, "output_dir"))
parser.add_argument('--experiment_tag', type=str, default='laptop')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--stride', type=int, default=4)
# parser.add_argument('--vit_f_dim', type=int, default=3025) # dino
parser.add_argument('--vit_f_dim', type=int, default=384) # dinov2
parser.add_argument("--model_type", type=str, default="dinov2_vits14")
# parser.add_argument('--res', type=int, default=112)
parser.add_argument('--res', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument("--annealing_lr", type=bool, default=True, help="Whether to use annealing learning rate.")
parser.add_argument('--batch_size_train', type=int, default=32, help='Batch size of the training dataloader.')
parser.add_argument('--batch_size_val', type=int, default=4, help='Batch size of the val dataloader.')
parser.add_argument("--checkpoint_model_path", type=str, default=None, help="Path to the model checkpoint.")
parser.add_argument('--data_path', type=str, default=os.path.join(file_dir, "d3dhoi_video_data"))
parser.add_argument('--data_load_ratio', type=float, default=0.2, help='The ratio of the data to be loaded as valid')
parser.add_argument('--save_every', type=int, default=200)
parser.add_argument('--val_every', type=int, default=50)
parser.add_argument('--continue_from_epoch', type=int, default=0)
parser.add_argument('--config_path', type=str, default=os.path.join(file_dir, "config"))
parser.add_argument('--SQ_template_path', type=str, default=os.path.join(file_dir, "SQ_templates"))
# parser.add_argument('--eval_images'

args = parser.parse_args()


def get_pred_leaf_rot6d(root_rot6d, leaf_rot_angle) -> torch.Tensor:
    R_root_to_leaf = rc.euler_angles_to_matrix(torch.Tensor([0, 0, leaf_rot_angle]).cuda(), 'XYZ')
    R_world_to_root = rc.rotation_6d_to_matrix(root_rot6d)
    R_world_to_leaf = torch.matmul(R_world_to_root, R_root_to_leaf)

    return R_world_to_leaf


def get_pred_leaf_trans(
    root_rot6d,
    root_trans,
    root_shape,
    root_size,
    root_rot_angle,
    leaf_rot_angle,
    leaf_shape,
    leaf_size,
) -> torch.Tensor:
    rotation_axis = torch.Tensor(
        [
            root_size[0] * (torch.cos(root_rot_angle) ** root_shape[1]),
            root_size[1] * (torch.sin(root_rot_angle) ** root_shape[1]),
            0,
        ]
    ).cuda()

    # Get vector point from rotation axis to leaf coordinate system origin
    alpha = torch.arctan(leaf_size[1] / leaf_size[0])
    rot_to_lef_len = torch.sqrt(
        (leaf_size[0] * (torch.cos(alpha) ** leaf_shape[1])) ** 2
        + (leaf_size[1] * (torch.sin(alpha) ** leaf_shape[1])) ** 2
    )
    rotation_to_leaf_sys = torch.Tensor(
        [
            torch.cos(alpha + leaf_rot_angle) * rot_to_lef_len,
            torch.sin(alpha + leaf_rot_angle) * rot_to_lef_len,
            0,
        ]
    ).cuda()

    # Origin of leaf coordinate system in root coordinate system
    leaf_origin_in_root_sys = rotation_axis + rotation_to_leaf_sys

    # Rotation matrix of root
    R_root = rc.rotation_6d_to_matrix(root_rot6d)

    world_to_leaf = torch.matmul(R_root, leaf_origin_in_root_sys) + root_trans

    return world_to_leaf


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


class OptimizerWrapper(object):
    def __init__(self, optimizer, aggregate=1):
        self.optimizer = optimizer
        self.aggregate = aggregate
        self._calls = 0

    def zero_grad(self):
        if self._calls == 0:
            self.optimizer.zero_grad()

    def step(self):
        self._calls += 1
        if self._calls == self.aggregate:
            self._calls = 0
            self.optimizer.step()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return OptimizerWrapper(
            torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                            weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "Adam":
        return OptimizerWrapper(
            torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "RAdam":
        return OptimizerWrapper(
            torch.optim.RAdam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    else:
        raise NotImplementedError()


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_checkpoints(epoch, model, optimizer, experiment_directory, args):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    # The optimizer is wrapped with an object implementing gradient
    # accumulation
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    if args.checkpoint_model_path is None:
        model_files = [
            f for f in os.listdir(experiment_directory)
            if f.startswith("model_")
        ]

        if len(model_files) == 0:
            return
        ids = [int(f[6:]) for f in model_files]
        max_id = max(ids)
        model_path = os.path.join(
            experiment_directory, "model_{:05d}"
        ).format(max_id)
        opt_path = os.path.join(experiment_directory, "opt_{:05d}").format(max_id)
        if not (os.path.exists(model_path) and os.path.exists(opt_path)):
            return

        print("Loading model checkpoint from {}".format(model_path))
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loading optimizer checkpoint from {}".format(opt_path))
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device)
        )
        args.continue_from_epoch = max_id+1
    else:
        print("Loading model checkpoint from {}".format(args.checkpoint_model_path))
        model.load_state_dict(torch.load(args.checkpoint_model_path, map_location=device))
        print("Loading optimizer checkpoint from {}".format(args.checkpoint_opt_path))
        optimizer.load_state_dict(
            torch.load(args.checkpoint_opt_path, map_location=device)
        )


def test():
    print ('To be finished...')


def train():
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d"%(args.gpu_id))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_tag))

    # Set log_dir for tensorboard
    log_dir = os.path.join(args.log_dir, experiment_tag)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    config = load_config(os.path.join(args.config_path, "tmp_config.yaml"))
    epochs = config["training"].get("epochs", 500)

    graphAE_params = Parameters()
    graphAE_params.read_config(os.path.join(args.config_path, "graphAE.config"))

    print("-----[Build NN]")
    net = Network_pts(
        test_mode=args.mode == 'test',
        device=device,
        stride=args.stride,
        model_type=args.model_type,
        vit_f_dim=args.vit_f_dim,
        hidden_dim=256
    ) # Finished TODO: Create the network
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # net.cuda()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.annealing_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Load the checkpoints if they exist in the experiment directory
    # load_checkpoints(net, optimizer, experiment_directory, args, device)

    # finished TODO: create the dataloader
    print("-----[Create Dataloader]-----")
    def collate_fn(batch):
        batch_rgb = [item["rgb"] for item in batch]
        batch_o_mask = [item["o_mask"] for item in batch]
        batch_info_3d = [item["info_3d"] for item in batch]
        batch_joint_state = [item["joint_state"] for item in batch]
        batch_image_names = [item["image_name"] for item in batch]

        # Convert rgb and o_mask to tensors
        batch_rgb = torch.stack(batch_rgb, dim=0)
        batch_o_mask = torch.stack(batch_o_mask, dim=0)

        return {
            "image_name": batch_image_names,
            "rgb": batch_rgb,
            "o_mask": batch_o_mask,
            "info_3d": batch_info_3d,
            "joint_state": batch_joint_state,
        }
    data_tag_path = os.path.join(args.data_path, args.experiment_tag)
    train_dataset = datasets.Datasets(datamat_path=data_tag_path, train=True, image_size=args.res, data_load_ratio=args.data_load_ratio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=1, drop_last=True, collate_fn=collate_fn)
    val_dataset = datasets.Datasets(datamat_path=data_tag_path, train=False, image_size=args.res, data_load_ratio=args.data_load_ratio)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=1, collate_fn=collate_fn)
    print ('Dataloader finished!')

    # TODO: create the differtiable renderer
    renderer = Nvdiffrast()
    print ('Renderer set!')

    print ('Start Training!')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.continue_from_epoch, epochs):
        net.train()

        total_loss = 0.
        iter_num = 0.
        for _, X in enumerate(tqdm(train_dataloader)):
            data_dict = X
            batch_size = data_dict['rgb'].shape[0]
            # Finished TODO: load all the data you need from dataloader, not limited
            image_names = data_dict['image_name']
            rgb_image = data_dict['rgb'].cuda()
            object_white_mask = data_dict['o_mask'].cuda()
            info3d = data_dict['info_3d']
            joint_state = data_dict['joint_state']
            height, width = 256, 256

            # Finished TODO: pass the input data to the network and generate the predictions
            pred_dict = net(
                rgb_image
            )
            pred_root_rot6d = pred_dict['pred_root_rot6d']
            pred_root_trans = pred_dict["pred_root_trans"] 
            pred_root_shape = pred_dict['pred_root_shape'] 
            pred_root_size = pred_dict['pred_root_size']
            pred_root_rot_angle = pred_dict["pred_root_rot_angle"] 
            pred_leaf_rot_angle = pred_dict["pred_leaf_rot_angle"] 
            pred_leaf_shape = pred_dict["pred_leaf_shape"]
            pred_leaf_size = pred_dict['pred_leaf_size']

            batch_rotation_matrix = rc.rotation_6d_to_matrix(pred_root_rot6d)

            batch_image = []
            for i in range(batch_size):
                # root object

                a1 = pred_root_size[i, 0]
                a2 = pred_root_size[i, 1]
                a3 = pred_root_size[i, 2]
                e1 = pred_root_shape[i, 0]
                e2 = pred_root_shape[i, 1]
                a1.retain_grad()
                R=batch_rotation_matrix[i]
                t=pred_root_trans[i]
                t = t.unsqueeze(1)
                points_root, faces =  points_on_sq_surface_torch(a1, a2, a3, e1, e2, R, t, n_samples=10)

                points_root.retain_grad()
                root_image = render(
                    points_root,
                    vertex_colors=torch.tensor([[[1, 0, 0]]], dtype=torch.float32).repeat(1, points_root.shape[0], 1).cuda(),
                    faces=faces,
                    resolution=(height, width),
                )
                # root_image = torch.pow(root_image, 1/2)
                root_image.retain_grad()

                # leaf object
                a1 = pred_leaf_size[i, 0, 0]
                a2 = pred_leaf_size[i, 0, 1]
                a3 = pred_leaf_size[i, 0, 2]
                e1 = pred_leaf_shape[i, 0, 0]
                e2 = pred_leaf_shape[i, 0, 1]

                # TODO
                R = get_pred_leaf_rot6d(pred_root_rot6d[i], pred_leaf_rot_angle[i])
                t = get_pred_leaf_trans(
                    root_rot6d=pred_root_rot6d[i],
                    root_trans=pred_root_trans[i],
                    root_rot_angle=pred_root_rot_angle[i],
                    root_shape=pred_root_shape[i],
                    root_size=pred_root_size[i],
                    leaf_rot_angle=pred_leaf_rot_angle[i, 0],
                    leaf_shape=pred_leaf_shape[i, 0],
                    leaf_size=pred_leaf_size[i, 0]
                )
                t = t.unsqueeze(1)
                points_leaf, faces_leaf = points_on_sq_surface_torch(a1, a2, a3, e1, e2, R, t, n_samples=10)
                leaf_image = render(
                    points_leaf,
                    vertex_colors=torch.tensor([[[1, 0, 0]]], dtype=torch.float32)
                    .repeat(1, points_leaf.shape[0], 1)
                    .cuda(),
                    faces=faces_leaf,
                    resolution=(height, width),
                )

                batch_image.append(root_image[0, :, :, 0] )

            batch_image = torch.stack(batch_image, dim=0)
            batch_image.retain_grad()
            loss = torch.nn.functional.mse_loss(batch_image, object_white_mask.float())

            # TODO: write the loss to tensorboard
            writer.add_scalar('train/loss', loss, epoch)

            total_loss += loss.item()

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            if args.annealing_lr:
                scheduler.step()
            optimizer.step()

        total_loss = float(total_loss) / float(iter_num)
        print ('[Epoch %d/%d] Total_loss = %f.' % (epoch, epochs, total_loss))

        if epoch % args.save_every == 0:
            save_checkpoints(
                epoch,
                net,
                optimizer,
                experiment_directory,
                args
            )

        if epoch % args.val_every == 0:
            print("====> Validation Epoch ====>")
            net.eval()

            total_eval_loss = 0.
            iter_num = 0.
            for imgi, X in enumerate(val_dataloader):
                # TODO: load data and generate the predictions, loss
                iter_num += 1

                data_dict = X

                batch_size = data_dict['rgb'].shape[0]
                # Finished TODO: load all the data you need from dataloader, not limited
                image_names = data_dict['image_name']
                rgb_image = data_dict['rgb'].cuda()
                object_white_mask = data_dict['o_mask'].cuda()
                info3d = data_dict['info_3d']
                joint_state = data_dict['joint_state']

                # Finished TODO: pass the input data to the network and generate the predictions
                pred_dict = net(
                    rgb_image
                )
                pred_root_rot6d = pred_dict['pred_root_rot6d']
                pred_root_trans = pred_dict["pred_root_trans"] 
                pred_root_shape = pred_dict['pred_root_shape'] 
                pred_root_size = pred_dict['pred_root_size']
                pred_root_rot_angle = pred_dict["pred_root_rot_angle"] 
                pred_leaf_rot_angle = pred_dict["pred_leaf_rot_angle"] 
                pred_leaf_shape = pred_dict["pred_leaf_shape"]
                pred_leaf_size = pred_dict['pred_leaf_size']

                batch_rotation_matrix = compute_rotation_matrix_from_ortho6d(pred_root_rot6d)

                batch_image = []

                for i in range(batch_size):
                    # root object
                    a1 = pred_root_size[i, 0]
                    a2 = pred_root_size[i, 1]
                    a3 = pred_root_size[i, 2]
                    e1 = pred_root_shape[i, 0]
                    e2 = pred_root_shape[i, 1]
                    R=batch_rotation_matrix[i]
                    t=pred_root_trans[i]
                    t = t.unsqueeze(1)
                    points_root, faces =  points_on_sq_surface_torch(a1, a2, a3, e1, e2, R, t, n_samples=20)
                    height, width = 256, 256
                    root_image = render(
                        points_root,
                        vertex_colors=torch.tensor([[[1, 0, 0]]], dtype=torch.float32)
                        .repeat(1, points_root.shape[0], 1)
                        .cuda(),
                        faces=faces,
                        resolution=(height, width),
                    )
                    # print(f"prediction for {image_names[i]}, \n    root a1: {a1}, a2: {a2}, a3: {a3}, e1: {e1}, e2: {e2}, R: {R}, t: {t}")

                    # leaf object
                    a1 = pred_leaf_size[i, 0, 0]
                    a2 = pred_leaf_size[i, 0, 1]
                    a3 = pred_leaf_size[i, 0, 2]
                    e1 = pred_leaf_shape[i, 0, 0]
                    e2 = pred_leaf_shape[i, 0, 1]
                    # print(
                    #     f"    leaf  a1: {a1}, a2: {a2}, a3: {a3}, e1: {e1}, e2: {e2}, R: {R}, t: {t}"
                    # )
                    # TODO
                    R = get_pred_leaf_rot6d(pred_root_rot6d[i], pred_leaf_rot_angle[i])
                    t = get_pred_leaf_trans(
                        root_rot6d=pred_root_rot6d[i],
                        root_trans=pred_root_trans[i],
                        root_rot_angle=pred_root_rot_angle[i],
                        root_shape=pred_root_shape[i],
                        root_size=pred_root_size[i],
                        leaf_rot_angle=pred_leaf_rot_angle[i, 0],
                        leaf_shape=pred_leaf_shape[i, 0],
                        leaf_size=pred_leaf_size[i, 0]
                    )
                    t = t.unsqueeze(1)
                    points_leaf, faces_leaf = points_on_sq_surface_torch(a1, a2, a3, e1, e2, R, t, n_samples=20)
                    leaf_image = render(
                        points_leaf,
                        vertex_colors=torch.tensor([[[1, 0, 0]]], dtype=torch.float32)
                        .repeat(1, points_leaf.shape[0], 1)
                        .cuda(),
                        faces=faces_leaf,
                        resolution=(height, width),
                    )

                    batch_image.append(root_image[0, :, :, 0])
                batch_image = torch.stack(batch_image, dim=0)
                loss = torch.nn.functional.mse_loss(batch_image,  object_white_mask.float())
                plt.imshow(batch_image[0, :, :].detach().cpu().numpy())
                plt.savefig("temp/test.pdf")
                plt.imshow(object_white_mask.float()[0, :, :].detach().cpu().numpy())
                plt.savefig("temp/test-gt.pdf")
                total_eval_loss += loss.item()

                if epoch % args.save_every == 0:
                    out_path = os.path.join(args.output_directory, experiment_tag)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    out_path = os.path.join(out_path, 'visualiza_results_epoch_%d' % (epoch))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # TODO: visualze the predicted results
            total_eval_loss = float(total_eval_loss) / float(iter_num) 
            writer.add_scalar("eval/loss", total_eval_loss, epoch)
            print("[Epoch %d/%d] Total_Eval_loss = %f." % (epoch, epochs, total_eval_loss))
            print("====> Validation Epoch ====>")

    print("Saved statistics in {}".format(experiment_tag))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print ('Bad Mode!')
        os._exit(0)
