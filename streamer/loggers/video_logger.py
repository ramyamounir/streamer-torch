import torch, math
import torch.nn as nn
import torch.nn.functional as F
from streamer.loggers.logger import Logger
from torchvision.utils import make_grid
from streamer.utils.logging import TBWriter
import matplotlib.cm as cm
import numpy as np

class VideoLogger(Logger):

    def model(self, prediction, groundtruth, attn_img, attns):
        self.counter += 1

        # checks
        if (self.log_base_every<=0) or \
           (not len(self.layer_logs)) or \
           (self.counter%self.log_base_every != 0):
               return

        # prediction logger
        if not hasattr(self, 'model_writer'):
            self.model_writer = TBWriter(self.writer, 'image', 'prediction images')

        if isinstance(prediction, torch.Tensor):
            # prediction = F.normalize(prediction, p=2, dim=1)
            imgs_list = [((prediction+1.0)/2.0).squeeze(0), ((groundtruth+1.0)/2.0).squeeze(0)]
            self.model_writer(make_grid([ *imgs_list ], nrow=2))


        # attention logger
        if not hasattr(self, 'model_attn'):
            self.model_attn = TBWriter(self.writer, 'image', 'attention images')

        if isinstance(attns, list):
            imgs_list = [((attn_img+1.0)/2.0).squeeze(0), self.visualize_attention(attns)]
            self.model_attn(make_grid([ *imgs_list ], nrow=2))


    @torch.no_grad()
    def visualize_attention(self, attention_maps):

        # Concatenate the attention maps from all blocks
        attention_maps = torch.cat(attention_maps, dim=1) # -> [1, blocks*heads, 65, 65]

        # select only the attention maps of the CLS token
        attention_maps = attention_maps[:, :, 0, 1:] # -> [1, blocks*heads, 64]

        # Then average the attention maps of the CLS token over all the heads
        attention_maps = attention_maps.mean(dim=1) # -> [1, 64]

        # Reshape the attention maps to a square
        num_patches = attention_maps.size(-1)
        size = int(math.sqrt(num_patches))
        attention_maps = attention_maps.view(-1, size, size) # -> [1, 8, 8]

        # Resize the map to the size of the image
        attention_maps = attention_maps.unsqueeze(1)
        attention_maps = F.interpolate(attention_maps, size=(128, 128), mode='bilinear', align_corners=False)
        attention_map = attention_maps.squeeze(1) # -> [1, 128, 128]

        # Normalize the attention map to range [0, 1]
        attention_map_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Convert the PyTorch tensor to a NumPy array
        attention_map_array = attention_map_normalized.cpu().numpy()

        # Apply the "jet" colormap
        cmap_jet = cm.get_cmap('jet')
        colored_attention_map = cmap_jet(attention_map_array)

        # Convert the RGBA image to RGB
        colored_attention_map_rgb = torch.Tensor(colored_attention_map[:, :, :, :3].tolist())[0].permute(2,0,1).cuda()
        return colored_attention_map_rgb

