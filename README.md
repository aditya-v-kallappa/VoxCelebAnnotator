## About
This repository provides a tool for automatic annotation of the VoxCeleb Dataset. The VoxCeleb Dataset is a large-scale speaker identification dataset containing speech data from celebrities obtained from YouTube videos. This tool performs Automatic Speech Recognition (ASR), Speaker Diarization, combines metadata, and presents a single annotated file, facilitating easier analysis and utilization of the dataset.

For more information about the VoxCeleb Dataset, visit [VoxCeleb Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

## Directory Structure
```
Root
├── txt
│   └── id10303
│       └── LKDWhdJQFco
│           └── 00001.txt
├── vox1_vox1_meta.csv
└── wav
    └── id10303
        └── LKDWhdJQFco
            └── 00001.wav
```

## Installation
```
git clone https://github.com/aditya-v-kallappa/VoxCelebAnnotator.git
cd VoxCelebAnnotator
pip install -r requirements.txt
```

See [here](https://github.com/m-bain/whisperX) for more information.
The code is tested with `cuda 12.2` on a `NVIDIA Tesla V100` GPU as well as on CPU 

## Usage 
To use the tool, run the `main.py` file with the required arguments (also some optional arguments).

- `'root_path'`: *str*, *required*, the path to the root of the train/dev/test dataset. It would be best if one runs `main.py` separately for train,dev and test datasets
- `'vox_metadata_path'`: *str*, the path to the metadata file from the dataset website. If not specified, the code will look for the metadata file in the `'root_path'` directory
- `'output_path'`: *str*, *required*, the path to the directory where annotation.jsonl, the final annotation file to be saved
- `'asr_model'`: *str*, A Whisper variant supported by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper/)
- `'device'`:  *str*, Device to run the pipeline on (cpu or cuda)
- `'compute_type'`: *str*, Precision to run whisper (int8 or float16 or float32)
- `'batch_size'`: *int*, Number of audio splits. Note that this is not the same as the number of audio files that are processed at once.
- `'fast_transcript'`: *bool*, Enabling this will not save probabilities and other information of transcription but will increase transcription speed 
- `'emotion_model'`: *str*, HF model to detect text based emotion. Default is `bhadresh-savani/bert-base-uncased-emotion`
- `'disable_asr'`: *bool*, This will disable ASR, Alignment and Emotion detection. The pipeline outputs audio details
- `'disable_alignemnt'`: *bool*, This will disable Alignment
- `'disable_emtion'`: *bool*, This will disable Emotion detection

To run the whole pipeline on GPU with half-precision, you need to run
    python main.py --root_path <root_path> --output_path <output_directory> --device cuda --compute_type float16


## Output JSONL file details 
THe final annotated file gets saved as a `jsonl` file. The description of the keys are given below.

- `'audio_details'`: *dict*, contains `audio_file_path`, `sampling_rate` and `length`
- `'transcript'`: *str*, contains the raw transcription of the audio
- `'transcript_info'`: *dict*, contians the output of `faster whisper` model. Contians various information about the transcription including `language`
- `'transcript_result_info'`: *str*, contains information about the decoding parameters
- `'alignment_results'`: *dict*, contians word to word alignment with speakers
- `'emotion'`:  *dict*, contains the probability scores of emotions detected by the emotion model
- `'label_info'`: *dict*, contains information about the audio that is given in the txt files of the dataset
- `'metadata'`: *dict*, contains information about the audio that is given in the metadata file of the dataset

### Acknowledgements
The ASR, alignment and diarization pipelines were possible because of [WhisperX](https://github.com/m-bain/whisperX). My gratitudes to them.

#### To do
- Multi-GPU support (atleast for the ASR pipeline)
- Optimize the code
- Better output formatting