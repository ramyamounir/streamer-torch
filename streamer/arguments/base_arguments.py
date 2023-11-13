import argparse

def bool_flag(s):
    r"""
    Parse boolean arguments from the command line.

    :param str s: command line string argument
    :returns bool: the boolean value of the argument
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_args():
    r"""
    parser function for cli arguments
    """

    parser = argparse.ArgumentParser(description='Streamer parser')

    # basic arguments
    parser.add_argument('--dataset', type=str, default='data/epic', help='Path to dataset folder')
    parser.add_argument('--output', type=str, default='out', help='Path to output folder')
    parser.add_argument('--type', type=str, choices=['streamer', 'dreamer'], default='streamer', help='model type')
    parser.add_argument('--name', type=str, default='my_model', help='name of the experiment')

    # Platform settings
    parser.add_argument('--p_name', type=str, default='job', help='platform job name for slurm')
    parser.add_argument('--p_device', type=str, choices=['gpu', 'slurm', 'cpu', 'mps'], default='gpu', help='platform device')
    parser.add_argument('--p_partition', type=str, default='general', help='platform partition with slurm')
    parser.add_argument('--p_n_nodes', type=int, default=1, help='platform number of nodes with slurm')
    parser.add_argument('--p_n_gpus', type=int, default=1, help='platform number of gpus')
    parser.add_argument('--p_n_cpus', type=int, default=2, help='platform number of total cpus per process')
    parser.add_argument('--p_ram', type=int, default=10, help='platform number of total ram in GB')
    parser.add_argument('--p_backend', type=str, choices=['nccl', 'gloo'], default='nccl', help='platform backend')
    parser.add_argument('--p_verbose', type=bool_flag, default=True, help='platform verbose')
    parser.add_argument('--p_logs', type=str, default='./logs', help='platform console logs')

    # dataset arguments
    parser.add_argument('--frame_size', type=int, nargs='+', default=[128,128], help='frame size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--dataset_split', type=str, default='train', choices=['train','test'], help='dataset split')
    parser.add_argument('--dataset_percent', type=int, default='100', help='dataset percentage to use')


    # model arguments
    parser.add_argument('--feature_dim', type=int, default=1024, help='feature dimension of each representation')
    parser.add_argument('--max_layers', type=int, default=3, help='maximum number of hierarchical layers')
    parser.add_argument('--evolve_every', type=int, default=1e5, help='add new layer every x steps')
    parser.add_argument('--init_layers', '-il', type=int, default=1, help='number of layers to initialize the model')
    parser.add_argument('--init_ckpt', '-ic', type=str, default='', help='ckpt path to load, overrides init_layers with layers count')
    parser.add_argument('--buffer_size', '-bs', type=int, default=10, help='buffer size for f function')
    parser.add_argument('--force_fixed_buffer', '-fb', type=bool_flag, default=False, help='force fixed buffer size')
    parser.add_argument('--l1_loss_weight', '-l1', type=float, default=100.0, help='l1 loss weight for dreamer')
    parser.add_argument('--demarcation_mode', type=str, default='average', choices=['fixed', 'accum', 'average'], help='Demarcation type')
    parser.add_argument('--distance_mode', type=str, default='similarity', choices=['similarity', 'distance'], help='distance type')
    parser.add_argument('--force_base_dist', type=bool_flag, default=False, help='force base layer distance mode to be distance')
    parser.add_argument('--normalize_imgs', type=bool_flag, default=False, help='normalize input images along channel dimension')
    parser.add_argument('--window_size', type=int, default=50, help='window size for average demarcation type')
    parser.add_argument('--modifier_type', type=str, default= 'multiply', choices=['add', 'multiply'], help='modifier type for average demarcation type')
    parser.add_argument('--modifier', type=float, default=1.0, help='modifier for average demarcation type')
    parser.add_argument('--loss_threshold', '-lt', type=float, default=0.1, help='loss threshold')

    # training arguments
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=3.0, help='reach hyperparameter for normalization')
    parser.add_argument('--optimize_every', type=int, default=100, help='optimize model every x steps of highest layer')
    parser.add_argument('--average_every', type=int, default=1000, help='average weights of models every x')
    parser.add_argument('--optimize', type=bool_flag, default=True, help='optimize the model')
    parser.add_argument('--save_every', type=int, default=25000, help='save model every x steps')
    parser.add_argument('--hgn_timescale', type=bool_flag, default=True, help='optimize higher layers with higher learning rate')
    parser.add_argument('--hgn_reach', type=bool_flag, default=True, help='optimize with reach of influence')
    parser.add_argument('--bp_up', type=bool_flag, default=True, help='backpropagate upwards to higher layers')
    parser.add_argument('--bp_down', type=bool_flag, default=True, help='backpropagate downwards to lower layers')
    parser.add_argument('--fp_up', type=bool_flag, default=True, help='forward propagate upwards to higher layers')
    parser.add_argument('--fp_down', type=bool_flag, default=True, help='forward propagate downwards to lower layers')

    # advanced arguments
    parser.add_argument('--dbg', action='store_true', help='Flag for debugging and development')
    parser.add_argument('--tb', action='store_true', help='Flag for tb logging')
    parser.add_argument('--log_prefix', type=str, default='/data/D2/datasets/epic_kitchen/videos/', help='prefix to write in json file')
    parser.add_argument('--log_postfix', type=str, default='MP4', help='postfix extension to write in json file')
    parser.add_argument('--log_base_every', type=int, default='1000', help='log base signal to tensorboard every')

    return parser.parse_args()



