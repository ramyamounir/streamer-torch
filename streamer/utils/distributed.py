import torch
import numpy as np, os, random
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from utils.logging import checkdir
import loggers


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_gpu(gpu, args):

    print(f'gpu {gpu} spawned', flush=True)

    # Initialize distributed environment
    args.gpu = gpu
    args.rank += gpu
    args.device = torch.device("cuda:{}".format(args.rank))

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    fix_random_seeds()
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    dist.barrier()

    args.main = (args.rank == 0)
    setup_for_distributed(args.main)

    # Tensorboard logger
    args.logger = None
    if args.tb:
        logger_class = loggers.getLogger(args)
        logger_path = os.path.join(args.exp_output, 'logs', str(args.rank))
        args.logger = logger_class(logger_path, args.log_base_every, args.distance_mode)

    args.json_dir = checkdir(f'{args.exp_output}/jsons/{args.rank}')



