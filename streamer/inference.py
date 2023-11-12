import argparse, json, os.path as osp
from models.inference_model import InferenceModel


def run_inference(args):

    # create inference model
    model = InferenceModel(checkpoint=args.weights)

    # run inference on video
    result = model(filename=args.input)


def parse_args():
    '''
    parser function for cli arguments
    '''
    parser = argparse.ArgumentParser(description='parser')

    # basic arguments
    parser.add_argument('--weights', '-w', type=str, required=True, help='Path to pretrained weights')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input file')

    return parser.parse_args()




if __name__ == "__main__":

    args = parse_args()
    run_inference(args)

