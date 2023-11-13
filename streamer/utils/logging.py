import torch
from torch.utils.tensorboard import SummaryWriter
import os, shutil, random, json, copy, cv2
import os.path as osp
import numpy as np
from skvideo.io import FFmpegWriter
from PIL import Image, ImageDraw, ImageFont
from collections import deque

def checkdir(path, reset = True):
    if os.path.exists(path):
        if reset:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

    return path


def setup_output(args):

    assert os.path.exists(args.dataset)
    assert (args.dbg) or (not os.path.exists(os.path.join(args.output, args.name))), 'experiment logs exist'

    # create output directory
    args.exp_output = checkdir(os.path.join(args.output, args.name))
    args.ckpt_dir = checkdir(f'{args.exp_output}/checkpoints')
    args.world_size = args.p_n_gpus * args.p_n_nodes

    # save commandline arguments and logs
    json.dump(args.__dict__, open(osp.join(args.exp_output, 'args.json'), 'w'), indent=2)
    args.p_logs = osp.join(args.exp_output, 'console_logs')

    return args


class TBWriter(object):

    def __init__(self, writer, data_type, tag, mul = 1, add = 0, fps = 4):

        self.step = 0
        self.mul = mul
        self.add = add
        self.fps = fps

        self.writer = writer
        self.type = data_type
        self.tag = tag

    def __call__(self, data, step = None, flush = False, metadata=None, label_img=None):

        counter = step if step != None else self.step*self.mul+self.add

        if self.type == 'scalar':
            self.writer.add_scalar(self.tag, data, global_step = counter)
        elif self.type == 'scalars':
            self.writer.add_scalars(self.tag, data, global_step = counter)
        elif self.type == 'image':
            self.writer.add_image(self.tag, data, global_step = counter)
        elif self.type == 'video':
            self.writer.add_video(self.tag, data, global_step = counter, fps = self.fps)
        elif self.type == 'figure':
            self.writer.add_figure(self.tag, data, global_step = counter)
        elif self.type == 'text':
            self.writer.add_text(self.tag, data, global_step = counter)
        elif self.type == 'histogram':
            self.writer.add_histogram(self.tag, data, global_step = counter)
        elif self.type == 'embedding':
            self.writer.add_embedding(mat=data, metadata=metadata, label_img=label_img, global_step=counter, tag=self.tag)

        self.step += 1

        if flush:
            self.writer.flush()


class JsonLogger():

    def __init__(self, 
                 is_inference = False,
                 snippet_size = 0.5,
                 prefix = '',
                 postfix = '',
                 json_dir =''
                 ):

        # arguments
        self.is_inference = is_inference
        self.snippet_size = snippet_size
        self.prefix = prefix
        self.postfix = postfix
        self.json_dir = json_dir


        self.json_template = dict(file="filename",
                             fileType = "video/mp4",
                             cursor = 0,
                             duration=100,
                             zoom=1,
                             layers=[])

        self.layer_template = dict(name = "layer 0",
                                   order = 0,
                                   annots = []
                                   )

        self.annot_template = dict(start=0.0,
                                   end=0.5,
                                   action="N/A",
                                   colour =f'rgb{0,0,0}',
                                   representation="null")

    def get_random_colors(self):
        result = []
        ind = random.randint(0, 2)
        for col in range(3):
            if col==ind: res = random.randint(75,100)
            else: res = random.randint(130, 250)
            result.append(res)

        return f'rgb({result[0]}, {result[1]}, {result[2]})'

    def get_colors(self, attn):
        if attn == None or attn == -1.0:
            return self.get_random_colors()
        
        result = (np.array([255, 255, 255]) * attn).astype(np.uint8).tolist()
        return f'rgb({result[0]}, {result[1]}, {result[2]})'

    def get_attn_norm(self, l):
        attn_list = [item/max(sublist) for sublist in l for item in sublist]
        return attn_list

    def create_layer_with_attn(self, boundaries, layer_num,  attention="null", representation="null"):

        # for edge cases where a new layer starts in the middle of processing a video
        if isinstance(attention, list) and len(attention)+1 < len(boundaries):
            attention = [-1.0 for  _ in range(len(boundaries)-len(attention)-1)] + attention

        if isinstance(representation, list) and len(representation)+1 < len(boundaries):
            representation = ["null" for  _ in range(len(boundaries)-len(representation)-1)] + representation


        layer_template = copy.deepcopy(self.layer_template)
        layer_template["name"] = f'layer {layer_num}'
        layer_template["order"] = layer_num

        for i in range(len(boundaries)-1):

            annot_template = copy.deepcopy(self.annot_template)
            annot_template["start"] = boundaries[i]
            annot_template["end"] = boundaries[i+1]
            annot_template["colour"] = self.get_colors(attention[i] if isinstance(attention, list) else None)
            annot_template["attn"] = attention[i] if isinstance(attention, list) else attention
            annot_template["representation"] = representation[i] if isinstance(representation, list) else representation
            layer_template["annots"].append(annot_template)


        return layer_template

    def create_layer(self, boundaries, layer_num):

        layer_template = copy.deepcopy(self.layer_template)
        layer_template["name"] = f'layer {layer_num}'
        layer_template["order"] = layer_num

        for i in range(len(boundaries)-1):

            annot_template = copy.deepcopy(self.annot_template)
            annot_template["start"] = boundaries[i]
            annot_template["end"] = boundaries[i+1]
            annot_template["colour"] = self.get_colors(attention[i] if isinstance(attention, list) else None)
            annot_template["attn"] = attention[i] if isinstance(attention, list) else attention
            annot_template["representation"] = representation[i] if isinstance(representation, list) else representation
            layer_template["annots"].append(annot_template)

        return layer_template

    def __call__(self, filepath, duration, hierarchy):


        h_boundaries = hierarchy['boundaries']

        output_file = copy.deepcopy(self.json_template)
        output_file["duration"] = duration

        if self.is_inference:
            output_file["file"] =  filepath
        else:
            output_file["file"] =  os.path.join(self.prefix, *filepath.split('/')[-2:]) + f'.{self.postfix}'


        for layer_num in range(len(h_boundaries)):

            layer_template = copy.deepcopy(self.layer_template)
            layer_template["name"] = f'layer {layer_num}'
            layer_template["order"] = layer_num

            boundaries = h_boundaries[layer_num]
            for i in range(len(boundaries)-1):
                annot_template = copy.deepcopy(self.annot_template)
                annot_template["start"] = boundaries[i]
                annot_template["end"] = boundaries[i+1]
                annot_template["colour"] = self.get_colors(attn=None)
                annot_template["action"] = "N/A"
                layer_template["annots"].append(annot_template)

            output_file["layers"].append(layer_template)

        if not self.is_inference:
            filename = os.path.splitext(os.path.basename(filepath))[0]
            json.dump(output_file, open(os.path.join(self.json_dir, filename+'.json'), 'w'))

        return output_file


    def log_with_attn(self, filepath, duration, hierarchy):

        h_boundaries = hierarchy['boundaries']
        h_attention = hierarchy['attention']

        output_file = copy.deepcopy(self.json_template)
        output_file["duration"] = duration

        if self.is_inference:
            output_file["file"] =  filepath
        else:
            output_file["file"] =  os.path.join(self.prefix, *filepath.split('/')[-2:]) + f'.{self.postfix}'


        for layer_num in range(len(h_boundaries)):
            at = self.get_attn_norm(h_attention[layer_num])

            if layer_num == 0:
                bs = np.linspace(start=0.0, stop=len(at)*self.snippet_size, num=len(at)+1, endpoint=True)
            else:
                bs = h_boundaries[layer_num-1]

            layer = self.create_layer(bs, layer_num, at)
            output_file["layers"].append(layer)


        layer = self.create_layer(h_boundaries[layer_num], layer_num+1)
        output_file["layers"].append(layer)

        if not self.is_inference:
            filename = os.path.splitext(os.path.basename(filepath))[0]
            json.dump(output_file, open(os.path.join(self.json_dir, filename+'.json'), 'w'))

        return output_file


class VideoWriterFFMPEG():
    def __init__(self, path, fps):
        self.video_writer = FFmpegWriter(path, inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})

    def __call__(self, frame):
        self.video_writer.writeFrame(frame)

    def __del__(self):
        self.video_writer.close()


class VideoWriterInference(VideoWriterFFMPEG):
    def __init__(self, path, fps):
        super(VideoWriterInference, self).__init__(path, fps)


        # Create a white background image
        self.width = 1280
        self.height = 500

        # Define the dimensions of small images
        self.small_width = 100
        self.small_height = 100
        self.small_padding = 25
        self.small_padding_y = 10

        # Define the dimensions of large images
        self.large_width = 300
        self.large_height = 300
        self.large_padding = 95
        self.large_padding_y = 10


        # Define text parameters
        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        self.font = ImageFont.truetype(font_path, size=64)
        self.text_color = (255, 0, 0)  # Red text color

        # avg buffer
        self.buffer = deque(maxlen=10)


    def __call__(self, buffer, prediction, groundtruth, text):


        # small images
        small_images = list(map(lambda x: (x.squeeze(0).permute(1,2,0).cpu().numpy()+1.0)/2.0, buffer))

        # buffer image
        # buffer_avg = np.mean(small_images, axis=0) if small_images else np.zeros((self.large_width,self.large_height,3))

        # prediction image
        prediction = (prediction.squeeze(0).permute(1,2,0).detach().cpu().numpy()+1.0)/2.0 if prediction != None else np.zeros((self.large_width,self.large_height,3))

        # groundtruth
        groundtruth = (groundtruth.squeeze(0).permute(1,2,0).cpu().numpy()+1.0)/2.0

        # buffer
        self.buffer.append(groundtruth)
        buffer_gt = np.mean(list(self.buffer), axis=0) if self.buffer else groundtruth

        # large images
        large_images = [buffer_gt, prediction, groundtruth]


        image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Paste small images in the first row
        x = self.small_padding
        y = 0
        for small_image in small_images:
            small_image = Image.fromarray((small_image*255.0).astype(np.uint8)).resize((self.small_width, self.small_height))
            image.paste(small_image, (x, y))
            x += self.small_width + self.small_padding

        # Paste large images in the second row
        x = self.large_padding
        y += (self.small_height+ self.small_padding_y)
        for large_image in large_images:
            large_image = Image.fromarray((large_image*255.0).astype(np.uint8)).resize((self.large_width, self.large_height))
            image.paste(large_image, (x, y))
            x += self.large_width + self.large_padding


        # Add text to the image
        text_width, text_height = draw.textsize(text, font=self.font)
        x = (self.width - text_width) // 2
        y += (self.large_height + self.large_padding_y)
        draw.text((x, y), text, fill=self.text_color, font=self.font)

        # write frame to video
        super(VideoWriterInference, self).__call__(np.array(image))




