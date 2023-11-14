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


def parser():
    r"""
    parser function for cli arguments
    """

    parser = argparse.ArgumentParser(description='Streamer parser')

    # basic arguments
    parser.add_argument('--dataset', type=str, default='data/epic', help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='out', help='Path to output directory. Will be created if does not exist')
    parser.add_argument('--name', type=str, default='my_model', help='Name of the experiment for logging purposes')

    # Platform settings
    parser.add_argument('--p_name', type=str, default='job', help='Platform job name for SLURM')
    parser.add_argument('--p_device', type=str, choices=['gpu', 'slurm', 'cpu', 'mps'], default='gpu', help='Platform device')
    parser.add_argument('--p_partition', type=str, default='general', help='Platform partition for SLURM')
    parser.add_argument('--p_n_nodes', type=int, default=1, help='Platform number of nodes for SLURM')
    parser.add_argument('--p_n_gpus', type=int, default=1, help='Platform number of GPUs per node')
    parser.add_argument('--p_n_cpus', type=int, default=2, help='Platform number of total CPUs per process/GPU')
    parser.add_argument('--p_ram', type=int, default=10, help='Platform total RAM in GB')
    parser.add_argument('--p_backend', type=str, choices=['nccl', 'gloo'], default='nccl', help='Platform backend for IPC')
    parser.add_argument('--p_verbose', type=bool_flag, default=True, help='Platform verbose')
    parser.add_argument('--p_logs', type=str, default='./logs', help='Platform console logs path. Will be added to output folder automatically')

    # dataset arguments
    parser.add_argument('--frame_size', type=int, nargs='+', default=[128,128], help='Frame size of images')
    parser.add_argument('--num_workers', type=int, default=1, help='Dataloader number of workers')
    parser.add_argument('--dataset_split', type=str, default='train', choices=['train','test'], help='Dataset split')
    parser.add_argument('--dataset_percent', type=int, default='100', help='Dataset portion to use in percentage')


    # model arguments
    parser.add_argument('--feature_dim', type=int, default=1024, help='Feature dimension of the representation at all layers')
    parser.add_argument('--max_layers', type=int, default=3, help='Maximum number of layers to stack')
    parser.add_argument('--evolve_every', type=int, default=1e5, help='Add new layer every *evolve_every* steps')
    parser.add_argument('--init_layers', type=int, default=1, help='Number of layers to initialize the model')
    parser.add_argument('--init_ckpt', type=str, default='', help='Checkpoint path to load. Overrides init_layers with layers count')
    parser.add_argument('--buffer_size', type=int, default=10, help='Buffer size for TemporalEncoding function')
    parser.add_argument('--force_fixed_buffer', type=bool_flag, default=False, help='Force fixed buffer size. If True, a boundary is forced after buffer is full')
    parser.add_argument('--demarcation_mode', type=str, default='average', choices=['fixed', 'accum', 'average'], help='Demarcation type. Only *average* is supported')
    parser.add_argument('--distance_mode', type=str, default='similarity', choices=['similarity', 'distance'], help='Distance type. Only *similarity* is supported')
    parser.add_argument('--force_base_dist', type=bool_flag, default=False, help='Force base layer distance mode to be *distance*. Not recommended')
    parser.add_argument('--normalize_imgs', type=bool_flag, default=False, help='Normalize input images along channel dimension')
    parser.add_argument('--window_size', type=int, default=50, help='Window size for *average* demarcation type')
    parser.add_argument('--modifier_type', type=str, default= 'multiply', choices=['add', 'multiply'], help='Modifier type for average demarcation type')
    parser.add_argument('--modifier', type=float, default=1.0, help='Modifier value for average demarcation type')
    parser.add_argument('--loss_threshold', type=float, default=0.1, help='Loss threshold for *fixed* and *accum* demarcation types.')

    # training arguments
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate of all modules and layers')
    parser.add_argument('--alpha', type=float, default=3.0, help='Reach hyperparameter for gradient normalization')
    parser.add_argument('--optimize_every', type=int, default=100, help='Optimize model every *optimize_every* steps of highest layer')
    parser.add_argument('--average_every', type=int, default=1000, help='Average weights of models every *average_every* steps of lowest layer')
    parser.add_argument('--optimize', type=bool_flag, default=True, help='Optimize the model. Set to False during inference or testing')
    parser.add_argument('--save_every', type=int, default=25000, help='Save model every *save_every* steps')
    parser.add_argument('--hgn_timescale', type=bool_flag, default=True, help='Optimize higher layers with bigger gradients')
    parser.add_argument('--hgn_reach', type=bool_flag, default=True, help='Optimize with reach of influence')

    # debugging and logging arguments
    parser.add_argument('--dbg', action='store_false', help='Flag for debugging and development. Overrides log files.')
    parser.add_argument('--tb', action='store_true', help='Flag for tb logging. If False, does not save tensorboard files.')
    parser.add_argument('--log_prefix', type=str, default='/data/D2/datasets/epic_kitchen/videos/', help='Prefix to write in json file')
    parser.add_argument('--log_postfix', type=str, default='MP4', help='Postfix extension to write in json file')
    parser.add_argument('--log_base_every', type=int, default='1000', help='Log base (e.g., images) signal to tensorboard every *log_base_every*')

    return parser
    # return parser.parse_args()



