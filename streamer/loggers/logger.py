import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from streamer.utils.logging import TBWriter, checkdir
from collections import deque
import random


class Logger():

    def __init__(self, writer_path, log_base_every, distance_mode):

        checkdir(writer_path)
        self.writer = SummaryWriter(writer_path)
        self.writer.flush()

        self.counter = 0
        self.log_base_every = log_base_every
        self.dist_mode = distance_mode
        self.layer_logs = []
        self.layer_deqs = []

        self.rep_deque = deque(maxlen=50)



    def model(self, prediction, groundtruth):
        pass

    def avg_rep(self, layer_num, representation):

        if len(self.layer_deqs) <= layer_num:
            self.layer_deqs.append(deque(maxlen=50))

        if random.random() > 0.5: self.layer_deqs[layer_num].append(representation.detach().clone())
        if len(self.layer_deqs[layer_num])<= 1: return 0.0

        embeddings = torch.cat(list(self.layer_deqs[layer_num]), dim=0)

        # needs to be fixed
        if self.dist_mode == 'distance':
            distances = torch.cdist(embeddings, embeddings, p=2)
        elif self.dist_mode == 'similarity':
            distances = torch.mm(embeddings, embeddings.t())
        else:
            raise Exception("distance mode not implemented")

        mask = torch.tril(torch.ones_like(distances), diagonal=-1).bool()
        val = distances[mask].mean()

        return val



    def layer(self, layer_num, boundary, loss, event_size, representation):
        if len(self.layer_logs) <= layer_num:
            self.layer_logs.append([
                TBWriter(self.writer, 'scalar', f'layer {layer_num}/loss {self.dist_mode}'),
                TBWriter(self.writer, 'scalar', f'layer {layer_num}/prediction length'),
                TBWriter(self.writer, 'scalar', f'layer {layer_num}/representation avg {self.dist_mode}'),]
                                   )



        self.layer_logs[layer_num][0](loss)
        if boundary: 
            self.layer_logs[layer_num][1](event_size)
            self.layer_logs[layer_num][2](self.avg_rep(layer_num, representation))

    def __del__(self):
        self.writer.close()


