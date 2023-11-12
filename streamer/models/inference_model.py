import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import VideoReader
import os, os.path as osp, importlib, cv2, json, copy
from streamer.utils.logging import JsonLogger, VideoWriterInference
from torchvision import transforms as T
from tqdm import tqdm
import streamer.models.networks as networks
from streamer.models.model import StreamerModel, StreamerModelArguments
from streamer.optimizers.streamer_optimizer import StreamerOptimizer, StreamerOptimizerArguments
from time import sleep
from torchvision.utils import make_grid
import streamer.optimizers

class InferenceLoader():
    r"""
    Dataloader used by InferenceModel to generate the images from the video

    :param str modality: only video is supported now
    :param str filename: Video filename to load
    :param float snippet_size: How many seconds per frame (typically 0.5)
    """

    def __init__(self, modality, filename, snippet_size):
        if modality =='video':
            self.transform = T.Compose([
                        T.Resize([128, 128], antialias=True),
                    ])
            self.filename = filename
            self.snippet_size = snippet_size

        elif modality == 'audio':
            raise NotImplementedError('Audio not implemented yet')

        self.init_file(self.filename)
    
    def init_file(self, file):
        self.reader = VideoReader(self.filename, 'video')
        self.metadata = self.get_metadata(self.reader)
        self.frame_interval = int(float(self.metadata['video']['fps'][0]) * self.snippet_size)
        self.num_frames = int(self.metadata['video']['fps'][0]*self.metadata['video']['duration'][0]//self.frame_interval)+1


    # fixed version from pytorch VideoReader
    def get_metadata(self, reader):
        """Returns video metadata

        Returns:
            (dict): dictionary containing duration and frame rate for every stream
        """
        if reader.backend == "pyav":
            metadata = {}  # type:  Dict[str, Any]

            for stream in reader.container.streams:
                if stream.type not in metadata:
                    if stream.type == "video":
                        rate_n = "fps"
                    else:
                        rate_n = "framerate"
                    metadata[stream.type] = {rate_n: [], "duration": []}

                rate = stream.average_rate if stream.average_rate is not None else stream.sample_rate

                try:
                    metadata[stream.type]["duration"].append(float(stream.duration * stream.time_base))
                    metadata[stream.type][rate_n].append(float(rate))
                except:
                    pass
            return metadata
        return reader._c.get_metadata()


    def __iter__(self):
        
        for frame_ix, frame in enumerate(self.reader):

            if frame_ix%self.frame_interval==0:
                frame_tens = (self.transform(frame['data']/255.0)*2.0 -1.0).unsqueeze(0)
                yield frame_tens



class InferenceModel():
    r"""
    Inference model used to run inference on a video using pretrained weights

    :param dict checkpoint: the loaded checkpoint of pretrained model
    :param Logger logger: tensorboard logger, if needed

    """

    def __init__(self, checkpoint, logger=None):


        self.checkpoint = checkpoint
        ckpt = torch.load(checkpoint)
        self.modality = ckpt['model_modality']
        self.snippet_size = ckpt['model_snippet_size']
        self.args = ckpt['model_args']
        self.logger = logger
        self.jsonlogger = JsonLogger(is_inference=True, snippet_size=self.snippet_size)

        self.model = self.get_model()

    def get_model(self):
        r"""
        Creates a streamer model and initializes it with ckpt.

        :returns:
            (:py:class:`~streamer.models.model.StreamerModel`): The initialized StreamerModel and moved to CUDA device

        """

        arguments = StreamerModelArguments(**self.args)
        arguments.init_ckpt = self.checkpoint

        if self.modality == 'video':
            model = StreamerModel(arguments, logger=self.logger, encoder=networks.CNNEncoder, decoder=networks.CNNDecoder)
        elif self.modality == 'audio':
            model = StreamerModel(arguments, logger=self.logger, encoder=None, decoder=None)

        model.modality = self.modality
        return model.cuda()


    def get_optimizer(self):
        r"""
        Creates an optimizer. Not needed in this implementation.

        :returns:
            (:py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer`): The optimizer to be used.

        """

        arguments = StreamerOptimizerArguments()
        optimizer = StreamerOptimizer(self.model, arguments)
        return optimizer


    def __call__(self, filename):
        r"""
        The function to run inference on a video and save logs.

        :param str filename: video path to run inference on.
        :returns:
            (*dict*): boundary results for all layers in a dictionary format
        """

        folder_path = osp.join(log_path,osp.splitext(osp.basename(filename))[0])
        os.makedirs(folder_path, exist_ok=True)

        loader = InferenceLoader(self.modality, filename, self.snippet_size)
        for frame in tqdm(loader, total=loader.num_frames):
            self.model(frame.cuda())

        hierarchy = self.model.extract_rep()
        json_file = self.jsonlogger(filename, loader.metadata['video']['duration'][0], hierarchy)

        # reset and close writer
        self.model.reset_model()
        del video_writer

        return json_file

