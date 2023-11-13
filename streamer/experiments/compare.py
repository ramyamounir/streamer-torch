import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from hlr import AnnotationsAsGraph


def compare(estimates, gt, gt_idx=0):
    van_files = [
        [], # estimates
        [] # gt
    ]

    for idx, t in enumerate([estimates, gt]):
        for root, _, files in os.walk(t):
            for file in files:
                filename = os.path.join(root, file)
                if filename.endswith('.json'):
                    van_files[idx].append(filename)


    if len(van_files[0]) != len(van_files[1]):
        print(f'Warning: you have {len(van_files[0])} prediction files and {len(van_files[1])} groundtruth files')


    # filter logic
    van_files_matches = []
    gt_basenames = [os.path.basename(p) for p in van_files[1]]
    for estimate in van_files[0]:
        if os.path.basename(estimate) in gt_basenames:
            gt_index = gt_basenames.index(os.path.basename(estimate))
            van_files_matches.append([estimate, van_files[1][gt_index]])



    iou, mof, denominator = 0, 0, 1e-9
    for e, g in tqdm(van_files_matches):

        ea = json.load(open(e, 'r'))
        ga = json.load(open(g, 'r'))

        gta = ga['layers'][gt_idx]
        ea['layers'].append(gta)

        _, (a, b) = AnnotationsAsGraph.run_file(ea, False)
        iou += a
        mof += b

        denominator += 1

    print(f'IoU: {iou/denominator}. \tMoF: {mof/denominator}')

def get_parser():
    parser = ArgumentParser(description='HLR comparison')

    parser.add_argument('--input', '-i', type=str, required=True, help='location of estimation JSON files')
    parser.add_argument('--ground_truth', '-gt', type=str, required=True, help='location of ground truth JSON files')
    parser.add_argument('--ground_truth_idx', '-idx', type=int, default=0, help='Level in GT to compare to')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()

    compare(args.input, args.ground_truth, args.ground_truth_idx)

