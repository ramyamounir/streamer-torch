import torch
import torch.nn as nn
import torch.nn.functional as F
from streamer.utils.events import CounterDetector
from streamer.utils.events import Patcher
from torch.utils.tensorboard import SummaryWriter
from streamer.utils.logging import TBWriter
from collections import deque
import streamer.models.networks as networks
from dataclasses import dataclass
from streamer.core.demarcation import EventDemarcation
from streamer.core.loss import StreamerLoss
from streamer.core.buffer import MemBuffer
import types

@dataclass
class StreamerLayerArguments():

    # architecture arguments
    max_layers: int
    r"""The maximum number of layers to stack"""

    feature_dim: int
    r"""Feature dimension of the model embeddings"""

    evolve_every: int
    r"""Create/stack a new layer every 'evolve_every' """

    buffer_size: int
    r"""Maximum input buffer size to be used"""

    loss_threshold: float
    r"""Loss threshold value. Not used in average demarcation mode"""

    lr: float
    r"""Learning rate to be used in all modules"""

    reps_fn: types.FunctionType 
    r"""Function to aggregate representations from all layers"""

    snippet_size: float
    r"""Snippet size of input video (seconds/image). Typically 0.5 seconds per image"""

    demarcation_mode: str = 'average'
    r"""Demarcation mode used to detect boundaries"""

    distance_mode: str = 'similarity'
    r"""Distance mode for loss calculation"""

    force_base_dist: bool = False
    r"""Force the lowest layer to use MSE instead of Cosine Similarity"""

    window_size: int = 50
    r"""Window size for average demarcation mode"""

    modifier_type: str = 'multiply'
    r"""Modifier type to apply to average demarcation mode ['multiply', 'add']"""

    modifier: float = 1.0
    r"""Modifier to apply to average demarcation mode"""

    force_fixed_buffer: bool = False
    r"""Force the buffer to be fixed (not replacing inputs) by triggering a boundary when buffer is full"""

    l1_loss_weight: float = 1.0
    r"""L1 loss weight. Not used in the current loss implementation"""

    @staticmethod
    def from_model_args(args):
        return StreamerLayerArguments(
                max_layers=args.max_layers,
                feature_dim=args.feature_dim,
                evolve_every=args.evolve_every,
                buffer_size=args.buffer_size,
                force_fixed_buffer=args.force_fixed_buffer,
                loss_threshold=args.loss_threshold,
                lr=args.lr,
                reps_fn=args.reps_fn,
                snippet_size=args.snippet_size,
                demarcation_mode=args.demarcation_mode,
                distance_mode=args.distance_mode,
                force_base_dist=args.force_base_dist,
                window_size=args.window_size,
                modifier_type=args.modifier_type,
                modifier=args.modifier,
                l1_loss_weight=args.l1_loss_weight,
        )

class StreamerLayer(nn.Module):
    r"""
    STREAMER layer implementation. 

    This layer can:
        * Create/stack other :py:class:`StreamerLayer` layers recursively for a maximum of :py:class:`~StreamerLayerArguments.max_layers`
        * Call the other :py:class:`StreamerLayer` layers by propagating current representation
        * Calculate and store the loss for the :py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer` to use it

    :param StreamerLayerArguments args: arguments provided to every streamer layer
    :param int layer_num: the index of the current layer in the layers stack
    :param int init_count: used to create more layers at initialization. Useful for Inference model using pretrained weights.
    :param torch.nn.Module encoder: Encoder class to be used at this layer. Passed later to the :py:class:`~streamer.models.networks.TemporalEncoding` module
    :param torch.nn.Module decoder: Decoder class to be used at this layer. Passed later to the :py:class:`~streamer.models.networks.HierarchicalPrediction` module
    :param Logger logger: Logger to be used for tensorboard.
    """

    def __init__(self, 
                 args, 
                 layer_num, 
                 init_count,
                 encoder=None, 
                 decoder=None,
                 logger=None
            ):

        super(StreamerLayer, self).__init__()

        # === References and Counters === #
        self.args = args
        self.logger = logger
        self.layer_num = layer_num
        self.init_count = init_count
        self.distance_mode = 'distance' if (args.force_base_dist and self.layer_num == 0) else args.distance_mode
        # self.global_distance_mode = args.distance_mode
        self.preprocess = (encoder != None)
        self.above = None
        self.layer_creator = CounterDetector(count=self.args.evolve_every)

        # === Memory Buffer === #
        self.buffer = MemBuffer(args.buffer_size, self.distance_mode)

        # === F function === #
        self.f = networks.TemporalEncoding(
                                feature_dim=args.feature_dim,
                                buffer_size=args.buffer_size,
                                lr=args.lr,
                                num_layers=2,
                                n_heads=8,
                                encoder=encoder,
                                patch=False,
                                )

        # === P function === #
        self.p = networks.HierarchicalPrediction(
                            feature_dim=args.feature_dim,
                            max_layers=args.max_layers,
                            lr=args.lr,
                            layer_num=self.layer_num,
                            num_layers=2,
                            n_heads=8,
                            decoder=decoder,
                            patch=False,
                            )

        # === Loss Function === #
        self.l = StreamerLoss(dist_mode = self.distance_mode)

        # === Demarcation Function === #
        self.demarcation = EventDemarcation(
                dem_mode = args.demarcation_mode,
                dist_mode = self.distance_mode,
                threshold = args.loss_threshold,
                window_size = args.window_size,
                modifier = args.modifier,
                modifier_type = args.modifier_type
                )



        # reset and clear
        self.reset_layer()
        print(f'created layer {self.layer_num}')

        # create more parent layers at init if needed
        self.create_parent(self.init_count>1)

    def reset_layer(self):
        r"""
        Reset function to be used at the beginning of a new video.
        Recursively applied to every layer.
        """

        self.buffer.reset_buffer()
        self.demarcation.reset()
        self.hierarchy_boundaries = [0]
        self.hierarchy_attn = []
        self.representation = None
        self.attn_img = None
        self.attns = None
        self.context = None
        self.objective_ready = False
        self.last_base_counter = -1
        if self.above != None: self.above.reset_layer()


    def extract_rep(self, hierarchy):
        r"""
        Extracts the hierarchy for json logging.
        Recursively applied to every layer.

        :param dict hierarchy: hierarchy dictionary to be filled with boundaries
        """

        hierarchy['boundaries'][self.layer_num] = self.hierarchy_boundaries.copy()
        if self.above != None: 
            self.above.extract_rep(hierarchy)

    def optimize_layer(self):
        r"""
        Optimization step function. Steps then zeros the gradients. 
        Calls the `step_params()` and `zero_grad()` functions of every module. (e.g., :py:meth:`~streamer.models.networks.TemporalEncoding.step_params`)
        Recursively applied to every layer.
        """

        # Step the optimizers
        self.f.step_params()
        self.p.step_params()
        self.l.step_params()

        # zero the gradients
        self.f.zero_grad()
        self.p.zero_grad()
        self.l.zero_grad()

        # reset representations because parameters have changed
        self.representation = None
        self.context = None

        if self.above != None:
            self.above.optimize_layer()

    def get_num_layers(self, num):
        r"""
        Recursive function to get the total number of layers

        :param int num: current number of layers at the previous layer
        :returns:
            (*int*): Previous num of layers + 1
        """
        if self.above != None:
            return self.above.get_num_layers(num+1)
        else: return num+1

    def create_parent(self, create):
        r"""
        Function to create/stack another :py:class:`StreamerLayer` layer

        :param bool create: only add another layer if create is True
        """
        if (not self.above) and (create) and (self.layer_num<(self.args.max_layers-1)) and (self.training):
            self.above = StreamerLayer(self.args, 
                                       self.layer_num+1, 
                                       self.init_count-1, 
                                       logger=self.logger).cuda()


    def predict(self):
        r"""
        Prediction function that calls the :py:class:`~streamer.models.networks.TemporalEncoding` and :py:class:`~streamer.models.networks.HierarchicalPrediction` modules
        """

        # === prepare input === #
        f_in = torch.cat(self.buffer.get_inputs(), dim=0)
        self.attn_img = f_in[0]

        # === Temporal Encoding === #
        '''
        input: [S, 3, 128, 128] or [S, 1024]
        outputs: [1, 1024]
        '''
        self.representation, self.attns = self.f(f_in)
        if self.distance_mode=="similarity": self.representation = F.normalize(self.representation, p=2, dim=-1)

        # === Hierarchical Prediction === #
        '''
        input: [L, 1024]
        outputs: [1, 3, 128, 128] or [1, 1024]
        '''
        reps = self.args.reps_fn(self.layer_num)
        self.context = self.p(reps)


    def forward(self, x, base_counter):
        r"""
        Forward propagation function for a layer. Recursively calls the layer above at event boundary determined by the :py:class:`~streamer.core.demarcation.EventDemarcation` module.

        :param torch.Tensor x: the input feature vector [1, feature_dim] or image [1, 3, H, W]
        :param int base_counter: the location of this input in the video for timescale caluation in the :py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer`
        """

        # create parent if needed
        self.create_parent(self.layer_creator())
        self.x_size = base_counter - self.last_base_counter
        self.last_base_counter = base_counter

        # check if layer is new or recently optimized
        if self.context == None:
            self.buffer.reset_buffer()
            self.buffer.add_input(x, 0.0)
            self.predict()
            return 

        # check loss of prediction with input
        self.main_objective, demarcation_signal = self.l(self.context, x)
        self.objective_ready = True

        # check boundary
        boundary_demarcation = self.demarcation(demarcation_signal.item())
        boundary_buffer =  self.args.force_fixed_buffer and self.buffer.counter >= self.buffer.buffer_size
        boundary = boundary_demarcation or boundary_buffer

        # Tensorboard logging
        if self.logger != None: 
            self.logger.layer(self.layer_num, 
                              boundary,
                              demarcation_signal.item(),
                              self.buffer.counter,
                              self.representation.detach())

        # new event created at this level, send current event upwards
        if boundary:

            # save boundary values for json
            self.hierarchy_boundaries.append(base_counter*self.args.snippet_size)

            # if above layer exists, send it up
            if self.above != None: self.above(self.representation.detach().clone(), base_counter=base_counter)
            self.buffer.reset_buffer()


        # represent and predict
        self.buffer.add_input(x, demarcation_signal.item())
        self.predict()

