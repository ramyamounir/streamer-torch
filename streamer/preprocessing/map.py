import os, argparse
import json
from glob import glob
from tqdm import tqdm


def get_epic_fps(vid_name):
    fps = 50.0 if len(vid_name.split('_')[1])==3 else 59.94
    return fps


def count_files_by_extension(root_path, target_extension):

    # find the first file to know depth and glob folders
    found = False
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(target_extension.lower()):
                print(f'found file: {os.path.join(root, file)}')
                folders = glob(os.path.join(root_path, *['*' for _ in range(len(root.lstrip(root_path).split('/')))]))
                found = True; break
            if found: break
        if found: break

    print(f'Example folder: {folders[0]}')

    # get number of files for every folder
    folder_info = {folder:sum(1 for _ in os.scandir(folder) if _.is_file()) for folder in tqdm(folders)}
    # folder_info = {folder:len(glob(os.path.join(folder, f'*.{target_extension}'))) for folder in tqdm(folders)}
    
    return folder_info


def map_dataset(args):

    mapped = dict(meta= dict(snippet_size=args.snippet_size, output_dim=args.output_dim),
                  files= dict())

    file_counts = count_files_by_extension(args.input, args.input_ext)

    for file_path, file_count in file_counts.items():

        if args.orig_fps:
            fps = get_epic_fps(os.path.basename(file_path))
            mapped["files"][file_path] = int(file_count//(fps*args.snippet_size))
        else:
            mapped["files"][file_path] = file_count

    with open(os.path.join(args.input, 'map.json'), 'w') as f:
        json.dump(mapped, f)


def parse_args():
    '''
    parser function for cli arguments
    '''

    parser = argparse.ArgumentParser(description='Epic Kitchens Preprocessing')

    # basic arguments paths and extensions
    parser.add_argument('--input', '-i', required=True, type=str, help='Path to input folder')
    parser.add_argument('--input_ext', '-ix', type=str, default='jpg', help='input file extension')

    # mapping arguments
    parser.add_argument('--snippet_size', '-ss', type=float, default=0.5, help='the snippet size in time')
    parser.add_argument('--output_dim', '-od', type=int, nargs='+', default=[128, 128], help='the feature dimension of each snippet')

    # flags
    parser.add_argument('--orig_fps', '-ofps', action='store_true', help='only for extracted epic at original fps')
    return parser.parse_args()


if __name__ == "__main__":

    # get arguments
    args = parse_args()

    # map the dataset to json
    map_dataset(args)

