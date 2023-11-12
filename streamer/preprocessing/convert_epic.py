import pandas as pd
import copy, random, os, json, argparse
from moviepy.editor import VideoFileClip
from tqdm import tqdm



class JsonLogger():

    def __init__(self, videos_path, jsons_path):

        self.videos_path = videos_path
        self.jsons_path = jsons_path

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
                                   colour =f'rgb{0,0,0}')

    def get_random_colors(self):
        result = []
        ind = random.randint(0, 2)
        for col in range(3):
            if col==ind: res = random.randint(75,100)
            else: res = random.randint(130, 250)
            result.append(res)

        return f'rgb({result[0]}, {result[1]}, {result[2]})'


    def time_to_seconds(self, time_str):
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s

    def __call__(self, v_id, df):

        part_id = v_id.split("_")[0]
        video_path = os.path.join(self.videos_path, part_id, v_id + '.MP4')
        video_duration = VideoFileClip(video_path).duration

        output_file = copy.deepcopy(self.json_template)
        output_file["duration"] = video_duration
        output_file["file"] =  video_path

        layer_template = copy.deepcopy(self.layer_template)
        layer_template["name"] = f'gt'
        layer_template["order"] = 0

        for index, row in df.iterrows():

            annot_template = copy.deepcopy(self.annot_template)
            annot_template["start"] = self.time_to_seconds(row["start_timestamp"])
            annot_template["end"] = self.time_to_seconds(row["stop_timestamp"])
            annot_template["colour"] = self.get_random_colors()
            annot_template["action"] = row["narration"]
            layer_template["annots"].append(annot_template)

        output_file["layers"].append(layer_template)

        save_path = os.path.join(self.jsons_path, part_id, v_id+'.json')

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        json.dump(output_file, open(save_path, 'w'))
        return output_file


def convert_dataset(args):

    # read csv
    df = pd.read_csv(args.input_annot)

    # select on the desired columns and group them by video
    df = df[["video_id", "start_timestamp", "stop_timestamp", "narration"]]
    grouped_df = df.groupby("video_id")

    # create json logger
    json_logger = JsonLogger(videos_path=args.input_videos, jsons_path=args.output_jsons)

    # get json and save it
    for v_id, group in tqdm(grouped_df):
        json_file = json_logger(v_id, group)




def parse_args():
    '''
    parser function for cli arguments
    '''

    parser = argparse.ArgumentParser(description='Epic Kitchens Preprocessing')

    # basic arguments paths and extensions
    parser.add_argument('--input_videos', '-iv', required=True, type=str, help='videos path')
    parser.add_argument('--input_annot', '-ia', required=True, type=str,  help='annotations csv file path')
    parser.add_argument('--output_jsons', '-oj', required=True, type=str,  help='output jsons path')
    return parser.parse_args()


if __name__ == "__main__":

    # get arguments
    args = parse_args()

    # map the dataset to json
    convert_dataset(args)


