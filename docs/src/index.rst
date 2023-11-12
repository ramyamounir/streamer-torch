STREAMER
########

**STREAMER** is a predictive learning model that uses continually train to improve its future predictions at different timescales.
It uses the prediction error to segment events in a hierarchical manner while processing streaming videos.

We provide code snippet in the API documentation on how to instantiate different classes.
We also provide simple training and inference scripts to reproduce the results in the `NeurIPS'23 paper <https://ramymounir.com/publications/streamer/>`_.


Training Script
===============


.. note::
    This training script uses commandline arguments as defined in the `code <https://github.com/ramyamounir/streamer-neurips23/blob/main/streamer/arguments/base_arguments.py>`_.
    Pretrained weights will be released soon..


.. code-block:: python
    :emphasize-lines: 18,21,43,46

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


    args = parse_args()
    args = setup_output(args)

    mp.spawn(train_gpu, args = (args,), nprocs = args.world_size)


====


Inference Script
================


.. code-block:: python

    from streamer.models.inference_model import InferenceModel

    model = InferenceModel(checkpoint='to/checkpoint/path/')
    result = model(filename='to/video/file/path')




.. toctree::
   :caption: Introduction
   :glob:
   :hidden:
   :titlesonly:

   quickstart/installation

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/core
   api/datasets
   api/optimizer
   api/layers
   api/models

.. toctree::
   :caption: Bureau
   :glob:
   :hidden:
   :titlesonly:

   LICENCE

