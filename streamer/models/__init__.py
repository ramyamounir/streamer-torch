import torch
import importlib, os
from streamer.models.networks import CNNEncoder, CNNDecoder
from streamer.models.model import StreamerModel, StreamerModelArguments
from streamer.models.inference_model import InferenceModel


def getModel(args):
    """
    Import the correct model
    """

    dataset_name = os.path.split(args.dataset.rstrip('/'))[-1]
    if dataset_name in ['epic', 'ego4d', 'kagu', 'finegym']:
        modality = "video"
    elif dataset_name in ['vox', 'timit']:
        modality = "audio"
    else:
        quit('dataset not found')

    arguments = StreamerModelArguments.from_args(args)

    if modality == 'video':
        model = StreamerModel(arguments, logger=args.logger, encoder=CNNEncoder, decoder=CNNDecoder)
    elif modality == 'audio':
        model = StreamerModel(arguments, logger=args.logger, encoder=None, decoder=None)

    model.modality = modality
    return model
    
