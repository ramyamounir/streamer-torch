import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from ddpw import Platform, Wrapper

from streamer.arguments.base_arguments import parse_args
from streamer.utils.distributed import init_gpu
from streamer.utils.logging import setup_output, JsonLogger

import streamer.datasets as datasets
import streamer.models as models
import streamer.optimizers as optimizers
from tqdm import tqdm



def train_gpu(global_rank, local_rank, args):

    # initialize gpu and tb writer, and return json logger
    init_gpu(global_rank, local_rank, args)

    # get dataloader instance
    loader = datasets.find_dataset_using_name(args)

    # get model instance
    model = models.getModel(args).to(args.device)

    # hierarchical gradient normalization and optimizer
    if args.optimize: 
        optimizer = optimizers.getOptimizer(args, model)

    # get logger
    jsonLogger = JsonLogger(is_inference = False,
                        prefix = args.log_prefix, 
                        postfix = args.log_postfix, 
                        snippet_size = args.snippet_size, 
                        json_dir = args.json_dir)


    # main training loop
    if args.main: loader = tqdm(loader)
    for batch_ix, (frames, info) in enumerate(loader):

        if args.normalize_imgs:
            frames = F.normalize(frames, dim=1, p=2)

        # forward and loss calculation
        model(frames.to(args.device))

        # backward and optimization
        if args.optimize: optimizer.step()

        # log video and reset if at the end
        if info[-1] == True:
            json_file = jsonLogger(filepath=info[0][0], 
                            duration=info[1].item(), 
                            hierarchy=model.extract_rep())
            model.reset_model()

        # save model every args.save_every
        if args.optimize and batch_ix % args.save_every == 0:
            model.save_model()

        # distributed barrier
        if args.world_size>1 and args.optimize: 
            dist.barrier()



    # save model every args.save_every
    if args.optimize and batch_ix % args.save_every == 0:
        model.save_model()

    if args.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if args.logger != None: del args.logger


if __name__ == "__main__":

    args = parse_args()
    args = setup_output(args)

    platform = Platform(
                    name=args.p_name,
                    device=args.p_device,
                    partition=args.p_partition,
                    n_nodes=args.p_n_nodes,
                    n_gpus=args.p_n_gpus,
                    n_cpus=args.p_n_cpus,
                    ram=args.p_ram,
                    backend=args.p_backend,
                    console_logs=args.p_logs,
                    verbose=args.p_verbose
                        )

    wrapper = Wrapper(platform=platform)

    # start training
    wrapper.start(train_gpu, args = args)

