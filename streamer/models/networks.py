import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    r"""
    A 4-layer CNN Encoder model used encode an image into a feature vector

    :param int feature_dim: the output feature dimension
    """

    def __init__(self, feature_dim):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 1024 , kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024 * 8 * 8, feature_dim)

    
    def forward(self, x):
        r"""
        The forward propagation function that takes input image and returns output vector

        :param torch.Tensor x: tensor of shape [1, 3, H, W]
        :returns:
            * (*torch.Tensor*): feature vector of shape [1, feature_dim]
            * (*None*): Used for compatibility to return attention in other models
        """

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 1024 * 8 * 8)
        x = self.fc1(x)
        return x, None


class CNNDecoder(nn.Module):
    r"""
    A 4-layer CNN Decoder model used decode a feature vector back to an image

    :param int feature_dim: the input feature dimension
    """

    def __init__(self, feature_dim):
        super(CNNDecoder, self).__init__()

        self.fc1 = nn.Linear(feature_dim, 1024 * 8 * 8)
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        r"""
        The forward propagation function that takes a feature vector and returns an image

        :param torch.Tensor x: tensor of shape [1, feature_dim]
        :returns:
            (*torch.Tensor*): image tensor of shape [1, 3, H, W]
        """

        x = self.fc1(x)
        x = x.view(-1, 1024, 8, 8)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2)
        x = self.conv4(x)
        return x


class TemporalEncoding(nn.Module):
    r"""
    The temporal encoding module receives as an input a sequence of feature vectors [S, feature_dim] or a sequence of images [S, 3, H, W]
    and returns the a single summary feature vector [1, feature_dim].

    If the input is a sequence of images, it instantiates a user-defined encoder class to encode the images into a series of feature vectors

    :param int feature_dim: the input feature dimension
    :param int buffer_size: the maximum buffer size to create positional encoding
    :param float lr: the learning rate of this module
    :param int n_heads: the number of heads for the attention layer
    :param int num_layers: the number of transformer encoder layers
    :param torch.nn.Module encoder: the encoder class to be used. Default: None
    :param bool patch: patch the transformer model to retain attention information. Default: False
    """


    def __init__(self, feature_dim, buffer_size, lr, n_heads=4, num_layers=2, encoder=None, patch=False):
        super(TemporalEncoding, self).__init__()

        if encoder != None: self.encoder = encoder(feature_dim)
        self.cls_token = nn.Embedding(1, feature_dim, max_norm=1.0)
        self.pos_enc = nn.Embedding(buffer_size, feature_dim, max_norm=1.0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=n_heads, dim_feedforward=feature_dim*2, activation=nn.ReLU(), dropout=0.0)
        norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)
        if patch: self.patcher = Patcher(self.network, patch=True) # patch and hook transformer

        # optimizer
        self.optim= torch.optim.Adam(self.parameters(), lr=lr)

    def get_params(self):
        r"""
        Function to extract the parameters of this module

        :returns:
            (*List(torch.tensor)*): List of parameters

        """
        return list(self.parameters())

    def get_grads(self, loss):
        r"""
        Computes the gradients with torch.autograd.grad.

        Not used in the current implementation.

        :returns:
            (*List(torch.tensor)*): List of gradients

        """
        if hasattr(self, 'optim'):
            return torch.autograd.grad(loss, self.optim.param_groups[0]['params'] , retain_graph=True, allow_unused=True)

    def step_params(self):
        r"""
        Applies gradient step on the parameters of this module. Called by :py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer`.
        """
        if hasattr(self, 'optim'): 
            self.optim.step()

    def zero_params(self):
        r"""
        Zeros out the gradients of the parameters of this module. Called by :py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer`.
        """
        if hasattr(self, 'optim'): 
            self.optim.zero_grad()



    def forward(self, x):
        r"""
        Forward propagation function that receives a sequence of inputs and returns a single feature vector summarizing the sequence.

        :param torch.Tensor x: an input a sequence of feature vectors [S, feature_dim] or a sequence of images [S, 3, H, W]

        :returns:
            * (*torch.tensor*): The output feature vector representation of the input sequence
            * (*torch.tensor* or *None*): The output attention values of the encoder. Return None if encoder not defined or if using CNNEncoder
        """

        # encode input if needed
        attns = None
        if hasattr(self, "encoder"): 
            x, attns = self.encoder(x) #[S,D]

        # foward pass through transformer
        # x = F.normalize(x, p=2, dim=-1)
        x_pos = x.unsqueeze(1) + self.pos_enc(torch.arange(x.shape[0]).cuda()).unsqueeze(1) #[S,1,D]
        x_cls = torch.cat([self.cls_token(torch.tensor([0]).cuda()).unsqueeze(0), x_pos], dim=0) #[S+1,1,D]
        representation = self.network(x_cls)[0]

        return representation, attns


class HierarchicalPrediction(nn.Module):
    r"""
    The hierarchical prediction modules receives as an input a sequence of feature vectors [S, feature_dim] where S is the number of layers
    and returns the a single prediction feature vector [1, feature_dim] or [1, 3, H, W] if a decoder class is provided.

    :param int feature_dim: the input feature dimension
    :param int max_layers: the maximum number of layers
    :param float lr: the learning rate of this module
    :param int layer_num: the layer number where this class is instantiated
    :param int n_heads: the number of heads for the attention layer
    :param int num_layers: the number of transformer encoder layers
    :param torch.nn.Module decoder: the decoder class to be used. Default: None
    :param bool patch: patch the transformer model to retain attention information. Default: False
    """

    def __init__(self, feature_dim, max_layers, lr, layer_num, n_heads=4, num_layers=2, decoder=None, patch=False):
        super(HierarchicalPrediction, self).__init__()

        self.layer_num = layer_num

        self.pos_enc = nn.Embedding(max_layers, feature_dim, max_norm=1.0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=feature_dim*2, activation=nn.ReLU(), dropout=0.0)
        norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)
        if decoder!= None: self.decoder = decoder(feature_dim)
        if patch: self.patcher = Patcher(self.network, patch=True) # patch and hook transformer

        # optimizer
        self.optim= torch.optim.Adam(self.parameters(), lr=lr)

    def get_params(self):
        r"""
        Function to extract the parameters of this module

        :returns:
            (*List(torch.tensor)*): List of parameters

        """
        return list(self.parameters())

    def get_grads(self, loss):
        r"""
        Computes the gradients with torch.autograd.grad.

        Not used in the current implementation.

        :returns:
            (*List(torch.tensor)*): List of gradients

        """
        if hasattr(self, 'optim'):
            return torch.autograd.grad(loss, self.optim.param_groups[0]['params'], retain_graph=True, allow_unused=True)

    def step_params(self):
        r"""
        Applies gradient step on the parameters of this module. Called by :py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer`.
        """
        if hasattr(self, 'optim'): 
            self.optim.step()

    def zero_params(self):
        r"""
        Zeros out the gradients of the parameters of this module. Called by :py:class:`~streamer.optimizers.streamer_optimizer.StreamerOptimizer`.
        """
        if hasattr(self, 'optim'): 
            self.optim.zero_grad()

    def forward(self, x):
        r"""
        Forward propagation function that receives a sequence of inputs and returns a single feature vector (or a single decoded image) predicting the next input.

        :param torch.Tensor x: an input a sequence of feature vectors [S, feature_dim]

        :returns:
            (*torch.tensor*): The output prediction feature vector [1, feature_dim] or decoded image [1, 3, H, W]
        """

        reps_pos = x.unsqueeze(1) + self.pos_enc(torch.arange(x.shape[0]).cuda()).unsqueeze(1)
        context = self.network(reps_pos)[self.layer_num]
        # context = F.normalize(context, p=2, dim=-1) 
        if hasattr(self, "decoder"): 
            context = self.decoder(context.unsqueeze(0))
            # context = F.normalize(context, p=2, dim=1) 

        return context


if __name__ == "__main__":

    # === Temporal Encoding === #

    encoding = TemporalEncoding(1024, 20, 1e-4, 4, 2, CNNEncoder).cuda()
    input = torch.randn(5,3,128,128).cuda()
    rep, attn = encoding(input)
    print(rep.shape)


    # === Hierarchical Prediction === #

    predictor = HierarchicalPrediction(1024, 3, 1e-4, 1, 4, 2, CNNDecoder).cuda()
    input = torch.randn(3,1024).cuda()
    context = predictor(input)
    print(context.shape)





