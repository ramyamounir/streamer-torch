# Preprocessing Dataset
This module contains files to preprocess datasets by extracting, storing statistics for training and converting groundtruth to unified json format similar to the output of STREAMER

---

### DATASET EXTRACTION

STREAMER architecture receives one input at a time. We preprocess the dataset to extract clips at some snippet size (e.g., 1 clip every 0.5 seconds).
The script `preprocessing/extract.py` preprocesses the video/audio dataset and saves the output to the desired directory as determined by the user.
Here are some arguments to use with the script:

- '--input', '-i': Path to the input dataset (e.g., '/datasets/epic_kitchens/', '/datasets/vox/').
- '--input_ext', '-ix': Input file extensions (e.g., 'MP4', 'ogg').
- '--output', '-o': Path to the output dataset (e.g., 'data/epic', 'data/vox').
- '--output_ext', '-ox': Input file extensions (e.g., 'jpg', 'txt').
- '--modality', '-m': Dataset modality. Choices are ['audio', 'video'].
- '--snippet_size', '-ss': Save one clip every *snippet_size* Seconds. (e.g., '0.5', '0.1')
- '--output_dim', '-od': output feature dimension.
    - accepts image resolution for modality videos (e.g., 128 128).
    - accepts number of values each snippet for modality audio (e.g., 1024).
- '--num_files', '-nf': How many files to process. Default=0 means all. (e.g., 10).

<details>
    <summary>Example Commands:</summary>

* To extract 1 frames every 0.5 seconds from 15 videos:

```bash
python preprocessing/extract.py --input /datasets/Ego4D/videos/ --input_ext MP4 --output data/ego --output_ext jpg --snippet_size 0.5 --output_dim 512 256 --num_files 15 --modality video
```

* To extract 1024 values every 0.1 seconds from 10 audio clips:

```bash
python preprocessing/extract.py --input /datasets/vox/ --input_ext ogg --output data/vox --output_ext txt --snippet_size 0.1 --output_dim 1024 --num_files 10 --modality audio
```
</details>


### DATASET MAPPING

After extracting the dataset into images/files, it is required to gather some statistics about the extracted script and save it to a json file. 
This is done in a separate script incase the dataset is already extracted and only needs to be mapped. 
The script `preprocessing/map.py` takes care of that. Here are some arguments:

- '--input', '-i': Path to the input dataset (e.g., 'data/epic/', '/data/vox/').
- '--input_ext', '-ix': Input file extensions (e.g., 'jpg', 'txt').
- '--snippet_size', '-ss': map one clip every *snippet_size* Seconds. (e.g., '0.5', '0.1')
- '--output_dim', '-od': output feature dimension.
    - accepts image resolution for modality videos (e.g., 128 128).
    - accepts number of values each snippet for modality audio (e.g., 1024).
- '--orig_fps', '-ofps': Only use for epic kitchen dataset that is already extracted with original fps of the video.

<details>
    <summary>Example Commands:</summary>

* To map epic dataset extracted with the extract.py script at 0.5 snippet_size:

```bash
python preprocessing/map.py -i data/epic -ix jpg -ss 0.5 -od 128 128
```

* To map epic dataset extracted at original fps and convert it to 0.5 snippet_size:

```bash
python preprocessing/map.py -i data/epic -ix jpg -ss 0.5 -od 128 128 -ofps
```
* To map vox dataset extracted at 0.1 snippet_size and feature dimension 1024:

```bash
python preprocessing/map.py -i data/vox -ix txt -ss 0.1 -od 1024
```

</details>


Example Dataset folder structure:

```bash
epic
├── map.json
├── P07
│   └── P07_111
│       ├── frame_0000000001.jpg
│       ├── frame_0000000002.jpg
│       └── frame_0000000003.jpg
└── P11
    └── P11_09
        ├── frame_0000000001.jpg
        ├── frame_0000000002.jpg
        └── frame_0000000003.jpg

```

---


### DATASET CONVERSION

After training the model, the predicted json files need to be compared with the groundtruth of the dataset.
The script `preprocessing/convert_epic.py` coverts the groundtruth of EPIC-KITCHENS to a unified json format to be evaluated against the model predictions. 
Here are some arguments:

- '--input_videos', '-iv': Path to the videos
- '--input_annots', '-ia': Path to the annotations csv file path
- '--output_jsons', '-oj': Path to the output jsons to be stored

<details>
    <summary>Example Commands:</summary>

* To convert the epic kitchens annotations

```bash
python datasets/convert_epic.py -iv data/epic/videos -ia data/epic/annots.csv -oj data/epic/groundtruth
```

</details>


