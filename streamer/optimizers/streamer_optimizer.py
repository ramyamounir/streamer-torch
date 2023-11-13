import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass


@dataclass
class StreamerOptimizerArguments():

    world_size: int = 1
    r"""Number of gpus to distribute the dataset"""

    alpha: int = 3
    r""" The reach parameter for Hierarchical Gradient Normalization"""

    max_layers: int = 3
    r"""The maximum number of layers to stack"""

    optimize_every: int = 100
    r"""Take a gradient step every this value"""

    average_every: int = 1000
    r"""Average models across gpus every this value"""

    hgn_timescale: bool = True
    r"""Allow timescale parameter in Hierarchical Gradient Normalization"""

    hgn_reach: bool = True
    r"""Allow reach parameter in Hierarchical Gradient Normalization"""

    bp_up: bool = True
    r"""Allow bottom up optimization"""

    bp_down: bool = True
    r"""Allow top down optimization"""

    @staticmethod
    def from_args(args):
        return StreamerOptimizerArguments(
                world_size=args.world_size, 
                alpha=args.alpha, 
                max_layers=args.max_layers, 
                optimize_every=args.optimize_every,
                average_every=args.average_every,
                hgn_timescale = args.hgn_timescale,
                hgn_reach = args.hgn_reach,
                bp_up = args.bp_up,
                bp_down = args.bp_down
        )



class StreamerOptimizer():
    r"""
    The optimizer used with streamer. 
    This class takes care of optimization, Gradient normalization and averaging across gpus.

    :param StreamerOptimizerArguments args: The parameters used for the Streamer optimizer
    """

    def __init__(self, model, args:StreamerOptimizerArguments):
        self.args = args
        self.model = model
        self.curr_n_layers = 0
        self.average_counter = 0
        self.step_counter = 0
        self.reset()

    def get_param_groups(self, layer_num):
        r"""
        Calculates the parameter groups and their weights.
        For example if the layer_num is 1 and we have 4 layers, then parameter groups will be [[1],[0,2],[3]] 
        and their weights will depend on alpha but typically more on the early groups (e.g., [0.8, 0.15, 0.05])

        :param int layer_num: the index of the layer
        :returns:
            * (*List(List(int))*): List of Lists dividing the layers into groups to assign different gradient multipliers to them
            * (*List(float)*): The weights assigned to the parameter groups
        """

        less = [layer_num - i for i in range(len(self.f_params)) if layer_num-i>=0]
        more = [layer_num + i for i in range(len(self.f_params)) if layer_num+i<len(self.f_params)]
        groups = []
        for i in range(max(len(less), len(more))):
            g = set()
            if i < len(less):
                g.add(less[i])
            if i < len(more):
                g.add(more[i])
            groups.append(list(g))


        weights = torch.Tensor([1.0/(self.args.alpha**i) for i in range(len(groups))])
        weights /= weights.sum()

        return groups, weights


    def get_current_params(self):
        r"""
        Gets a reference to all parameters of all layers in the streamer model
        """

        n_layers = 1
        ll = self.model.streamer
        self.f_params = [ll.f.get_params()]
        self.p_params = [ll.p.get_params()]
        self.l_params = [ll.l.get_params()]

        while ll.above != None:
            n_layers += 1
            ll = ll.above
            self.f_params.append(ll.f.get_params())
            self.p_params.append(ll.p.get_params())
            self.l_params.append(ll.l.get_params())

        return n_layers

    def check_bp(self, layer, curr_layer):
        up, down = (layer < curr_layer), (layer > curr_layer)
        if not (up or down): return True
        if up and self.args.bp_up: return True
        if down and self.args.bp_down: return True
        return False


    def update_all_layers(self, layer):

        # return if no loss to use
        if not layer.objective_ready:
            return

        layer_num = layer.layer_num
        layer_loss = layer.l.summarize_loss(layer.main_objective)
        layer_timescale = float(layer.x_size) if self.args.hgn_timescale else 1.0

        self.p_counter[layer_num] += layer_timescale
        groups, reaches = self.get_param_groups(layer_num)

        for g_i, (group, reach) in enumerate(zip(groups, reaches)):

            # if not self.check_bp(layer_num, i): continue # need to be fixed

            # get parameters to accumulate
            params = [*self.p_params[layer_num]]
            for g_i in group: params.extend(self.f_params[g_i])

            retain_graph = g_i!=(len(groups)-1) or True
            (layer_loss*layer_timescale*reach).backward(inputs=params, retain_graph=retain_graph)
            for i in group: self.f_counter[i] += layer_timescale

        # reset loss
        self.step_counter[layer_num] += 1
        layer.objective_ready = False


    def get_gradients(self):
        r"""
        accumulates gradient on all layers' parameters from all losses in the model
        """

        ll = self.model.streamer
        self.update_all_layers(ll)

        while ll.above != None:
            ll = ll.above
            self.update_all_layers(ll)


    def equal_layers(self):
        r"""
        Determines if the streamer model has the same number of layers on all gpus

        :returns:
            * (*bool*): True if equal layers on all gpus
            * (*int*): The number of layers on the current gpu
        """
        l_global = torch.tensor([0.0]).cuda()
        l_global = self.model.streamer.get_num_layers(l_global)
        l_local = l_global.clone()

        if self.args.world_size == 1:
            return True, int(l_local.item())

        dist.all_reduce(l_global)
        return l_local*self.args.world_size == l_global, int(l_local.item())


    def reset(self):
        r"""
        Resets the counters of the optimizer
        """

        self.f_counter = {i:0 for i in range(self.args.max_layers)}
        self.p_counter = {i:0 for i in range(self.args.max_layers)}
        self.l_counter = {i:0 for i in range(self.args.max_layers)}

        self.step_counter = {i:0 for i in range(self.args.max_layers)}

    def scale_gradients(self):
        r"""
        Scales the gradients of all modules by the counters to normalize the gradients
        """

        def scale_module(params, counter):
            for p_ix, param in enumerate(params):
                if counter[p_ix] == 0: continue
                for p in param:
                    if p.grad != None: p.grad /= counter[p_ix]

        scale_module(self.f_params, self.f_counter)
        scale_module(self.p_params, self.p_counter)



    def average_models(self):
        r"""
        Average the model parameters across all gpus every :py:meth:`~StreamerOptimizerArguments.average_every`
        """

        for name, param in self.model.named_parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.args.world_size



    def step(self):
        r"""
        Call the optimizer, which calulates the gradients and accumulates it on the parameters.
        This function does not actually do gradient stepping. It has a counter that does it every :py:meth:`~StreamerOptimizerArguments.optimize_every`
        """

        # === AVERAGE MODELS === #
        if self.args.world_size > 1: self.average_counter += 1

        equal, n_layers = self.equal_layers()
        if n_layers > self.curr_n_layers: 
            self.curr_n_layers = self.get_current_params()

        if equal and self.average_counter >= self.args.average_every:
            self.average_models()
            self.average_counter = 0


        # === OPTIMIZATION === #
        self.get_gradients() # backwards
        if self.step_counter[self.curr_n_layers-1] >= self.args.optimize_every:
            self.scale_gradients()
            self.model.optimize_model()
            self.reset()





