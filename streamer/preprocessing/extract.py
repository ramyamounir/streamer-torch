'''
Copyright (c) Ramy Mounir

This file preprocesses a video dataset by extracting frames at the desired fps and rescales frames to the desired resolution. 
'''

import os, shutil, argparse
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle
from functools import partial
import multiprocessing as mp
import signal
import subprocess



def checkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def video_command(args, video_input, video_output):

    video_output = os.path.join(video_output, f'frame_%10d.{args.output_ext}')
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    # checkdir(os.path.dirname(video_output))

    command = f'\
        ffmpeg -i {video_input}\
        -vf "fps={int(1.0/args.snippet_size)},\
        scale={args.output_dim[0]}:{args.output_dim[1]}"\
        -vsync vfr {video_output}\
        -loglevel panic\
        > /dev/null 2>&1'

    process = subprocess.Popen(command, shell=True)
    process.communicate()


def audio_command(args, audio_input, audio_output):

    # checkdir(audio_output)
    os.makedirs(audio_output, exist_ok=True)

    audio, sr = librosa.load(audio_input, sr=args.output_dim[0]/args.snippet_size)
    num_snippets = len(audio)//args.output_dim[0]
    snippets = np.array_split(audio[:num_snippets * args.output_dim[0]], num_snippets)

    for idx, snippet in enumerate(snippets):
        output_file = os.path.join(audio_output, f'frame_{str(idx).zfill(10)}.{args.output_ext}')
        snippet_values = "\n".join(map(str, snippet))
        open(output_file, 'w').write(snippet_values)

def chunk_list_with_numpy(input_list, num_chunks):
    """Chunks a list using NumPy's array_split."""
    input_array = np.array(input_list)
    chunks = np.array_split(input_array, num_chunks)
    chunked_list = [chunk.tolist() for chunk in chunks]
    return chunked_list

def parallel_command(p_id, args, clips):

    progress_bar = tqdm(total=len(clips), desc=f'Process {p_id}', position=p_id)

    for file_input in clips:
        # file_output = os.path.join(args.output, file_input[len(args.input):-len(args.input_ext)-1].lstrip('/'))
        file_output = os.path.join(args.output, os.path.splitext(file_input[len(args.input):])[0].lstrip('/'))

        if args.modality == 'video':
            video_command(args, file_input, file_output)
        elif args.modality == 'audio':
            audio_command(args, file_input, file_output)

        progress_bar.update(1)

    # close bar
    progress_bar.close()

def find_files_by_extension(root_path, target_extension):
    file_list = []
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_extension = os.path.splitext(file.lower())[1].lstrip(".")
            if file_extension in target_extension:
            # if file.lower().endswith(target_extension.lower()):
                file_list.append(os.path.join(root, file))
                
    return file_list


def extract_clips(args):

    clips = find_files_by_extension(args.input, args.input_ext)
    print(f'Example file: {clips[0]} from {len(clips)} files')

    # get all files
    shuffle(clips)
    if args.num_files > 0: clips = clips[:args.num_files]

    # chunk to cpus
    clips_chunked = chunk_list_with_numpy(clips, args.num_cpus)

    processes = []
    for process in range(args.num_cpus):
        j = mp.Process(target=parallel_command, args=(process, args,clips_chunked[process]), daemon=True)
        processes.append(j)
        j.start()

    [p.join() for p in processes]


def parse_args():
    '''
    parser function for cli arguments
    '''

    parser = argparse.ArgumentParser(description='Epic Kitchens Preprocessing')

    # basic arguments paths and extensions
    parser.add_argument('--input', '-i', required=True, type=str, help='Path to input folder')
    parser.add_argument('--input_ext', '-ix', type=str, nargs='+', help='input extension')
    parser.add_argument('--output', '-o', required=True, type=str, help='Path to output folder')
    parser.add_argument('--output_ext', '-ox', type=str, default='txt', help='output files extension')

    # preprocessing arguments
    parser.add_argument('--modality', '-m', type=str, default='video', choices=['video', 'audio'], help='modality to process')
    parser.add_argument('--snippet_size', '-ss', type=float, default=0.1, help='the snippet size in time')
    parser.add_argument('--output_dim', '-od', type=int, nargs='+', default=1024, help='the feature dimension of each snippet')
    parser.add_argument('--num_files', '-nf', type=int, default=0, help='number of random files to process, 0 means all videos')

    # multiprocessing arguments
    parser.add_argument('--num_cpus', '-nc', type=int, default=0, help='number of cpus to use for multiprocessing, 0 means all cpus')

    # other arguments
    parser.add_argument('--dbg', action='store_true', help='Used to replace existing dataset')
    return parser.parse_args()


if __name__ == "__main__":

    # get arguments and reset dataset
    args = parse_args()
    assert ((args.dbg) or (not os.path.exists(args.output))), "Dataset folder already exists"
    checkdir(args.output)

    # set cpus
    available_cpus = os.sched_getaffinity(os.getpid())

    if args.num_cpus <= 0:
        args.num_cpus = len(available_cpus)
    else:
        assert args.num_cpus <= len(available_cpus), f"Maximum available cpus is {len(available_cpus)}"
        allowed_cpus = np.random.choice(list(available_cpus), size=args.num_cpus, replace=False).tolist()
        os.sched_setaffinity(os.getpid(), allowed_cpus)

    # extract frames if not map only
    extract_clips(args)

