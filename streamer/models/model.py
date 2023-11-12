import torch, os
import torch.nn as nn
import torch.nn.functional as F
from streamer.models.layer import StreamerLayer, StreamerLayerArguments
from dataclasses import dataclass, asdict

@dataclass
class StreamerModelArguments():

    log_base_every: int = 1000
    r"""Tensorboard log of the base layer every 'log_base_every'"""

    main: bool = True
    r"""Main process/gpu or not"""

    max_layers: int = 3
    r"""The maximum number of layers to stack"""

    feature_dim: int = 1024
    r"""Feature dimension of the model embeddings"""

    evolve_every: int = 50000
    r"""Create/stack a new layer every 'evolve_every' """

    buffer_size: int = 10
    r"""Maximum input buffer size to be used"""

    force_fixed_buffer: bool = False
    r"""Force the buffer to be fixed (not replacing inputs) by triggering a boundary when buffer is full"""

    loss_threshold: float = 0.25
    r"""Loss threshold value. Not used in average demarcation mode"""

    lr: float = 1e-4
    r"""Learning rate to be used in all modules"""

    init_layers: int = 1
    r"""How many layers to initialize before training"""

    init_ckpt: str = ''
    r"""the path of the pretrained weights, if any"""

    ckpt_dir: str = ''
    r"""the path to save weights"""

    snippet_size: float = 0.5
    r"""Snippet size of input video (seconds/image). Typically 0.5 seconds per image"""

    demarcation_mode: str = 'average'
    r"""Demarcation mode used to detect boundaries"""

    distance_mode: str = 'distance'
    r"""Distance mode for loss calculation"""

    force_base_dist: bool = True
    r"""Force the lowest layer to use MSE instead of Cosine Similarity"""

    window_size: int = 50
    r"""Window size for average demarcation mode"""

    modifier_type: str = 'multiply'
    r"""Modifier type to apply to average demarcation mode ['multiply', 'add']"""

    modifier: float = 1.0
    r"""Modifier to apply to avrrage demarcation mode"""

    fp_up: bool = True
    r"""Allow bottom up inference"""

    fp_down: bool = True
    r"""Allow top down inference"""

    l1_loss_weight: float = 100
    r"""L1 loss weight. Not used in the current loss implementation"""


    @staticmethod
    def from_args(args):
        return StreamerModelArguments(
                init_ckpt=args.init_ckpt,
                init_layers=args.init_layers,
                ckpt_dir=args.ckpt_dir,
                snippet_size=args.snippet_size,
                log_base_every=args.log_base_every,
                main=args.main,
                max_layers=args.max_layers,
                feature_dim=args.feature_dim,
                evolve_every=args.evolve_every,
                buffer_size=args.buffer_size,
                force_fixed_buffer=args.force_fixed_buffer,
                loss_threshold=args.loss_threshold,
                l1_loss_weight=args.l1_loss_weight,
                lr=args.lr,
                demarcation_mode=args.demarcation_mode,
                distance_mode=args.distance_mode,
                force_base_dist=args.force_base_dist,
                window_size=args.window_size,
                modifier_type=args.modifier_type,
                modifier=args.modifier,
                fp_up=args.fp_up,
                fp_down=args.fp_down
        )


    @staticmethod
    def from_ckpt_args(args):
        return StreamerModelArguments(
                init_ckpt=args.init_ckpt,
                init_layers=args.init_layers,
                ckpt_dir=args.ckpt_dir,
                snippet_size=args.snippet_size,
                log_base_every=args.log_base_every,
                main=args.main,
                max_layers=args.max_layers,
                feature_dim=args.feature_dim,
                evolve_every=args.evolve_every,
                buffer_size=args.buffer_size,
                force_fixed_buffer=args.force_fixed_buffer,
                loss_threshold=args.loss_threshold,
                l1_loss_weight=args.l1_loss_weight,
                lr=args.lr,
                demarcation_mode=args.demarcation_mode,
                distance_mode=args.distance_mode,
                force_base_dist=args.force_base_dist,
                window_size=args.window_size,
                modifier_type=args.modifier_type,
                modifier=args.modifier,
                fp_up=args.fp_up,
                fp_down=args.fp_down
        )



class StreamerModel(nn.Module):
    r"""
    The implementation of the STREAMER model for training. This class initializes the first layer(s) of streamer, saves/loads weights, etc.

    :param StreamerModelArguments args: The arguments passed to Streamer Model
    :param Logger logger: The tensorboard logger class
    :param torch.nn.Module encoder: The encoder model (e.g., :py:class:`~streamer.models.networks.CNNEncoder`)
    :param torch.nn.Module decoder: The decoder model (e.g., :py:class:`~streamer.models.networks.CNNDecoder`)
    """

    def __init__(self, args:StreamerModelArguments, logger=None, encoder=None, decoder=None):
        super(StreamerModel, self).__init__()
        self.args = args
        self.logger = logger

        self.streamer = None
        self.models_saved = 0

        # counters
        self.base_counter = 0
        self.__initialize(args.init_ckpt, args.init_layers, encoder=encoder, decoder=decoder)



    def init_layer(self, count=1, encoder=None, decoder=None):
        r"""
        Initializes the Streamer layer(s) and passes the enocder/decoder to the first layer.
        
        :param int count: How many layers to create 
        :param torch.nn.Module encoder: The encoder model (e.g., :py:class:`~streamer.models.networks.CNNEncoder`)
        :param torch.nn.Module decoder: The decoder model (e.g., :py:class:`~streamer.models.networks.CNNDecoder`)
        """

        # Arguments
        self.args.reps_fn = self.getReps
        streamerLayerArgs = StreamerLayerArguments.from_model_args(self.args)

        self.streamer = StreamerLayer(
                args = self.args,
                layer_num = 0, 
                init_count = count,
                encoder=encoder,
                decoder=decoder,
                logger=self.logger
                )


    def __initialize(self, ckpt='', count=1, encoder=None, decoder=None):
        r"""
        Initializes the Streamer layer(s) with checkpoint if available.
        
        :param str ckpt: pretrained weights location
        :param int count: How many layers to create 
        :param torch.nn.Module encoder: The encoder model (e.g., :py:class:`~streamer.models.networks.CNNEncoder`)
        :param torch.nn.Module decoder: The decoder model (e.g., :py:class:`~streamer.models.networks.CNNDecoder`)
        """

        if ckpt=='':
            self.init_layer(count=count, encoder=encoder, decoder=decoder)

        else:
            ckpt= torch.load(ckpt, map_location='cpu')
            self.init_layer(count=ckpt['num_layers'], encoder=encoder, decoder=decoder)
            self.streamer.load_state_dict(ckpt['weights'])
            

    def forward(self, x):
        r"""
        Forward propagation function that calls the :py:meth:`~streamer.models.layer.StreamerLayer.forward` function of the first :py:class:`~streamer.models.layer.StreamerLayer`.
        
        :param torch.Tensor x: the input image [1, 3, H, W]
        """

        # log base signal
        if self.logger != None: 
            self.logger.model(self.streamer.context, 
                              x, 
                              self.streamer.attn_img, 
                              self.streamer.attns)

        # main forward
        self.streamer(x, base_counter=self.base_counter)
        self.base_counter += 1


    def getReps(self, layer_num):
        r"""
        Aggregates the representations from all the layers.

        :param int layer_num: the index of the calling layer

        :returns:
            (*torch.Tensor*): concatenated representations from all layers [L, feature_dim]

        """

        def check_fp(layer, curr_layer):
            down, up = (layer < curr_layer), (layer > curr_layer)

            if not (up or down): return True
            if up and self.args.fp_up: return True
            if down and self.args.fp_down: return True
            return False

        curr_layer_num = 0
        ll = self.streamer
        reps = [ll.representation] if check_fp(layer_num, curr_layer_num) else [ll.representation.detach()]

        while ll.above != None and ll.above.representation != None:
            curr_layer_num += 1
            ll = ll.above
            if check_fp(layer_num, curr_layer_num): reps.append(ll.representation)
            else: reps.append(ll.representation.detach())

        return torch.cat(reps, dim=0)

    def extract_rep(self):
        r"""
        Extract representation function used for logging json hierarchy. 
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.extract_rep` on the :py:class:`~streamer.models.layer.StreamerLayer` class

        :returns:
            (*dict*): hierarchy represented as boundaries of every layer

        """

        if self.streamer == None:
            return {}

        hierarchy = dict(boundaries = {})
        self.streamer.extract_rep(hierarchy)

        # close open boundaries
        for layer_name, layer_bounds in hierarchy['boundaries'].items():
            if layer_name == 0: continue
            if layer_bounds[-1] != hierarchy['boundaries'][0][-1]:
                layer_bounds.append(hierarchy['boundaries'][0][-1])

        return hierarchy

    def reset_model(self):
        r"""
        Resets the whole streamer model for a new video.
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.reset_layer` on the :py:class:`~streamer.models.layer.StreamerLayer` class

        """
        if self.streamer == None:
            return 

        self.streamer.reset_layer()
        self.base_counter = 0

    def get_num_layers(self):
        r"""
        Get the total number of layers.
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.get_num_layers` on the :py:class:`~streamer.models.layer.StreamerLayer` class
        """
        if self.streamer == None:
            return 0

        return self.streamer.get_num_layers(0)

    def optimize_model(self):
        r"""
        Optimizes the whole streamer model (gradient step). 
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.optimize_layer` on the :py:class:`~streamer.models.layer.StreamerLayer` class
        """
        if self.streamer != None:
            self.streamer.optimize_layer()
        

    def save_model(self):
        r"""
        Saves the model weights to :py:class:`~StreamerModelArguments.ckpt_dir`
        """

        if self.streamer == None or self.args.main == False:
            return 

        ckpt = dict(
                model_args = asdict(self.args),
                model_modality = self.modality,
                model_snippet_size = self.args.snippet_size,
                num_layers = self.get_num_layers(), 
                weights = self.streamer.state_dict())

        torch.save(ckpt, os.path.join(self.args.ckpt_dir, f'model_{str(self.models_saved).zfill(3)}.pth'))
        self.models_saved += 1

