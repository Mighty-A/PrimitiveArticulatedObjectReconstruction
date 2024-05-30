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
from plyfile import PlyData

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from renderer_nvdiff import Nvdiffrast
from networks.baseline_network import Network_pts
from myutils.graphAE_param import Parameters

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
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--annealing_lr", type=bool, default=True, help="Whether to use annealing learning rate.")
parser.add_argument('--batch_size_train', type=int, default=8, help='Batch size of the training dataloader.')
parser.add_argument('--batch_size_val', type=int, default=1, help='Batch size of the val dataloader.')
parser.add_argument("--checkpoint_model_path", type=str, default=None, help="Path to the model checkpoint.")
parser.add_argument('--data_path', type=str, default=os.path.join(file_dir, "d3dhoi_video_data"))
parser.add_argument('--data_load_ratio', type=float, default=0.2, help='The ratio of the data to be loaded as valid')
parser.add_argument('--save_every', type=int, default=200)
parser.add_argument('--val_every', type=int, default=200)
parser.add_argument('--continue_from_epoch', type=int, default=0)
parser.add_argument('--config_path', type=str, default=os.path.join(file_dir, "config"))
parser.add_argument('--SQ_template_path', type=str, default=os.path.join(file_dir, "SQ_templates"))
# parser.add_argument('--eval_images'

args = parser.parse_args()


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


def load_init_template_data(path):
    data_path_dict = scipy.io.loadmat(
                                      )
    data_dict = {}

    init_object_sq = np.load(data_path_dict['object_init_sq_path'][0])
    data_dict['init_object_sq'] = torch.Tensor(init_object_sq).cuda()
    init_object_rots = np.load(data_path_dict['object_init_rots_path'][0])
    data_dict['init_object_rots'] = torch.Tensor(init_object_rots).cuda()
    init_object_old_center = np.load(data_path_dict['object_init_old_center_path'][0])
    data_dict['init_object_old_center'] = torch.Tensor(init_object_old_center).cuda()

    data_dict['object_axle'] = data_path_dict['object_axle'][0]
    data_dict['object_num_bones'] = data_path_dict['object_num_bones'][0][0]

    return data_dict

def load_init_SQ_template_data(path):
    joint_info_path = os.path.join(path, 'joint_info.mat')
    part_centers_path = os.path.join(path, 'part_centers.npy')
    ply_path = os.path.join(path, 'plys')
    delta_rots_path =  os.path.join(ply_path, 'delta_rots.npy')
    pred_rots_path = os.path.join(ply_path, "pred_rots.npy")
    pre_sq_path = os.path.join(ply_path, "pred_sq.npy")

    data_dict = {}

    joint_info = scipy.io.loadmat(joint_info_path)
    data_dict['joint_info'] = joint_info
    data_dict["init_object_old_center"] = np.load(part_centers_path)
    data_dict['object_joint_tree'] = joint_info['joint_tree']
    data_dict['object_primitive_align'] = joint_info['primitive_align']
    data_dict['object_joint_parameter_leaf'] = joint_info['joint_parameter_leaf']
    data_dict['object_num_bones'] = joint_info['joint_tree'].shape[1]

    pts = []
    for i in range(data_dict["object_num_bones"]):
        ply_data = PlyData.read(os.path.join(ply_path, "SQ_ply", f"{i}.ply"))

        pts.append(ply_data['vertex'])

    data_dict['object_input_pts'] = pts

    return data_dict


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
        graphAE_param=graphAE_params,
        test_mode=args.mode == 'test',
        device=device,
        stride=args.stride,
        model_type=args.model_type,
        vit_f_dim=args.vit_f_dim
    ) # Finished TODO: Create the network
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    
    # net.cuda()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.annealing_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(net, optimizer, experiment_directory, args, device)

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
    # init_template_data_dict = load_init_template_data(args.data_path)
    init_template_data_dict = load_init_SQ_template_data(os.path.join(args.SQ_template_path, args.experiment_tag))
    print ('Dataloader finished!')

    # TODO: create the differtiable renderer
    renderer = Nvdiffrast(FOV=39.6)
    print ('Renderer set!')

    print ('Start Training!')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.continue_from_epoch, epochs):
        net.train()

        total_loss = 0.
        iter_num = 0.
        for _, X in enumerate(tqdm(train_dataloader)):
            data_dict = X

            # TODO: load all the data you need from dataloader, not limited
            image_names = data_dict['image_name']
            rgb_image = data_dict['rgb'].cuda()
            object_white_mask = data_dict['o_mask'].cuda()
            info3d = data_dict['info_3d']
            joint_state = data_dict['joint_state']
            object_input_pts = init_template_data_dict["object_input_pts"]
            init_object_old_center = init_template_data_dict["init_object_old_center"]
            object_num_bones = init_template_data_dict['object_num_bones']
            object_joint_tree = init_template_data_dict['object_joint_tree']
            object_primitive_align = init_template_data_dict["object_primitive_align"]
            object_joint_parameter_leaf = init_template_data_dict["object_joint_parameter_leaf"]
            cam_trans = None # seems useless
            # TODO: pass the input data to the network and generate the predictions
            pred_dict = net(
                rgb_image,
                object_input_pts,
                init_object_old_center,
                object_num_bones,
                object_joint_tree,
                object_primitive_align,
                object_joint_parameter_leaf,
                cam_trans
            )

            print(pred_dict)
            image = renderer(pred_dict["deformed_object"])

            # TODO: compute loss functions
            loss = torch.nn.functional.mse_loss(image, object_white_mask)

            # TODO: write the loss to tensorboard
            writer.add_scalar('train/loss', loss, epoch)

            total_loss += loss.item()

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
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

                pred_dict = net(...)

                if epoch % args.save_every == 0:
                    out_path = os.path.join(args.output_directory, experiment_tag)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    out_path = os.path.join(out_path, 'visualiza_results_epoch_%d' % (epoch))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # TODO: visualze the predicted results

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
