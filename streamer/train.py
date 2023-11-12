import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import datasets, models, optimizers
from arguments.base_arguments import parse_args
from utils.distributed import init_gpu
from utils.logging import setup_output, JsonLogger
from tqdm import tqdm


def train_gpu(gpu, args):

    # initialize gpu and tb writer, and return json logger
    init_gpu(gpu, args)

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
        if args.optimize: dist.barrier()


    # save model every args.save_every
    if args.optimize and batch_ix % args.save_every == 0:
        model.save_model()

    dist.barrier()
    dist.destroy_process_group()

    if args.logger != None: del args.logger


if __name__ == "__main__":

    args = parse_args()
    args = setup_output(args)

    mp.spawn(train_gpu, args = (args,), nprocs = args.world_size)

