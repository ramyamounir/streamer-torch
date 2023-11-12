import torch.nn as nn

class StreamerLoss(nn.Module):
    r"""
    Streamer loss function is applied to minimize the distance between the prediction and actual input.
    Implemented for Cosine Similarity and Euclidean distance.
    Use only Cosine Similarity as it backpropagates bounded loss values across layers.

    :param str dist_mode: The distance mode: ['similarity', 'distance']
    """

    def __init__(self, dist_mode):
        super(StreamerLoss, self).__init__()
        self.dist_mode = dist_mode
        self.criterion = nn.CosineSimilarity() if dist_mode=='similarity' else nn.MSELoss()


    def forward(self, context, groundtruth):
        r"""
        Calculates the loss value from context and groundtruth

        :param torch.Tensor context: The prediction tensor. shape=[1,3,H,W] or [1,feature_dim]
        :param torch.Tensor groundtruth: The groundtruth tensor. shape=[1,3,H,W] or [1,feature_dim]
        :returns: 
            * (*dict*): dictionary of all the losses
            * (*float*): the loss value to be used for demarcation (i.e., input to the :py:meth:`~streamer.core.demarcation.EventDemarcation.__call__`)
        """

        loss = self.criterion(context, groundtruth).mean()
        losses_dict = dict(l2_loss = loss)
        return losses_dict, loss

    def get_params(self): return []
    def step_params(self): pass
    def zero_params(self): pass

    def summarize_loss(self, loss_dict):
        r"""
        Converts losses_dict from the output of :py:meth:`.forward` to loss value respecting the distance mode.
        Called by the optimizer

        :param dict losses_dict: The full losses dictionary
        :returns:
            (*float*): The loss value to be minimized
        """
        loss_sign = -1 if self.dist_mode == 'similarity' else 1
        return loss_sign * sum(list(loss_dict.values()))



