import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterDetector:
    def __init__(self, count):
        self.count = count
        self.counter = 0

    def __call__(self):

        if self.counter > self.count:
            return True
        else:
            self.counter += 1

        return False

class CycleDetector:
    def __init__(self, count):
        self.count = count
        self.counter = 1

    def __call__(self):

        if self.counter == self.count:
            self.counter = 1
            return True
        else:
            self.counter += 1
            return False


class PlateauDetector:
    def __init__(self, window_size=10, threshold=1e-6):
        self.window_size = window_size
        self.threshold = threshold
        self.values = []
        self.mean = None

    def __call__(self, value):
        self.values.append(value)

        if len(self.values) <= self.window_size:
            return False

        self.values.pop(0)
        mean = torch.tensor(self.values).mean()

        if (value>=0.5) and (self.mean is not None) and (math.isclose(mean, self.mean, rel_tol=self.threshold)):
            return True

        self.mean = mean

        return False


class Patcher():

    def __init__(self, model, patch=True):
        self.reset()
        self.num_layers = len(model.layers)

        for l in model.layers:
            if patch: self.patch_layer(l.self_attn)
            l.self_attn.register_forward_hook(self.save_output)

    def patch_layer(self, m):

        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True

            return forward_orig(*args, **kwargs)

        m.forward = wrap

    def save_output(self, m, m_in, m_out):
        if len(self.outputs) == self.num_layers:
            self.reset()

        self.outputs.append( m_out[1].mean(dim=0) )

    def calculate_attn(self, index=0):
        if self.attention_calculated: return None

        self.outputs.reverse()

        self.final_attention = self.outputs[0]
        for attn in self.outputs[1:]:
            self.final_attention = self.final_attention@attn

        self.attention_calculated = True

        attn_lst = self.final_attention[index][1:]
        attn_lst /= sum(attn_lst)
        return attn_lst.tolist()

    def reset(self):
        self.outputs = []
        self.final_attention= None
        self.attention_calculated = False


