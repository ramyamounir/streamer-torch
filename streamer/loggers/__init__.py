import torch
import importlib, os


def getLogger(args):

    """
    Import the correct logger
    """

    dataset_name = os.path.split(args.dataset.rstrip('/'))[-1]
    if dataset_name in ['epic', 'ego4d', 'kagu', 'finegym']:
        modality = "video"
    elif dataset_name in ['vox', 'timit']:
        modality = "audio"
    else:
        quit('dataset not found')

    logger_filename  = f'streamer.loggers.{modality}_logger'
    modellib = importlib.import_module(logger_filename)

    logger_class = None
    target_logger_name =  f'{modality}Logger'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_logger_name.lower():
            logger_class = cls

    if logger_class is None:
        raise NotImplementedError("In %s.py, there should be a class name that matches %s in lowercase." % (logger_filename, target_logger_name))

    return logger_class
    


