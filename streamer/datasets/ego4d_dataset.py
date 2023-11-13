import torch.nn.functional as F
import os, json, cv2, random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataclasses import dataclass


@dataclass
class Ego4dDatasetArguments():
    r"""
    Arguments for Ego4d dataset
    """

    world_size: int
    r"""Number of gpus to distribute the dataset"""

    global_rank: int
    r"""The rank of the running device"""

    dataset: str
    r"""The dataset name"""

    frame_size: list
    r"""The frame size to which the images will be resized"""

    percentage: int
    r"""Percentage of the dataset to run"""

    split: str = 'train'
    r"""The split of data to use. Choices: ['train', 'test']"""

    @staticmethod
    def from_args(args):
        return Ego4dDatasetArguments(
                world_size=args.world_size, 
                global_rank=args.global_rank, 
                dataset=args.dataset, 
                frame_size=args.frame_size,
                percentage=args.dataset_percent,
                split=args.dataset_split
        )


class Ego4dDataset(Dataset):
    r"""
    The ego4d dataset that iterates over the whole dataset and returns the frames one by one in a streaming fashion.

    :param Ego4dDatasetArguments args: The parameters used for the Ego4d dataset
    """

    def __init__(self, args: Ego4dDatasetArguments):

        vids_map_path = os.path.join(args.dataset, 'map.json')
        if not os.path.exists(vids_map_path):
            quit('mapping file does not exist')

        with open(vids_map_path, 'r') as f:
            vid_paths_dict_full = json.load(f)
            self.snippet_size = vid_paths_dict_full['meta']['snippet_size']
            self.vids_paths = vid_paths_dict_full['files']

        # sample a percentage of files
        self.vids_paths = dict(random.sample(self.vids_paths.items(), int(len(self.vids_paths)*args.percentage/100.0)))
        print(f'processing {len(self.vids_paths)} for the {args.split} split')


        self.min_sum, self.gpu_train_paths = self.split_dict_greedy(
                {k:[v, v] for k,v in self.vids_paths.items()}, 
                args.world_size,
                args.global_rank)

        self.prepare_data()
        self.len = len(self.data)

        self.transform = T.Compose([
                    T.ToTensor(),
                    T.Resize(args.frame_size, antialias=True),
                ])

    def prepare_data(self):

        to_go = self.min_sum

        self.data = []
        for video in self.gpu_train_paths:
            img_paths = np.array(sorted(glob(f'{video}/*.jpg')))
            duration = len(img_paths)*self.snippet_size

            for i in range(len(img_paths)):
                self.data.append([video, duration, i, i*self.snippet_size, False])
                to_go -= 1
                if to_go == 0: break

            self.data[-1][-1] = True
            if to_go == 0: break


    # Split array into chunks using greedy. Courtesy of Bing AI
    def split_dict_greedy(self, dct, n, r):

        # Get the keys and values from the dictionary and sort them by values in descending order
        keys_values = list(zip(dct.keys(), dct.values()))
        keys_values.sort(key=lambda x: x[1][1], reverse=True)
        keys, lst = zip(*keys_values)

        # Convert keys and lst into lists
        keys = list(keys)
        lst = list(lst)

        # Initialize empty chunks and sums
        chunks = [[] for _ in range(n)]
        sums = [0] * n

        # Loop through the list and assign each element to the chunk with the smallest sum
        for i in range(len(lst)):
            min_index = sums.index(min(sums))
            chunks[min_index].append(keys[i])
            sums[min_index] += lst[i][1]

        return min(sums), chunks[r]


    def __getitem__(self, index):
        r"""
        Iterates over the dataset in a streaming fashion and retrieves one frame at a time

        :param int index: The index of the item in the dataset to retrieve
        :returns:
            * (*torch.tensor*): the frame in tensor format
            * (*List*): Information about the frame [Video, duration, index, time, last_frame]
        """

        info = self.data[index]
        frames = self.transform(cv2.cvtColor(cv2.imread(f'{info[0]}/{str(info[2]+1).zfill(8)}.jpg'), cv2.COLOR_BGR2RGB))*2.0-1.0
        return frames, info

    def __len__(self):
        return self.len
