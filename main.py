import argparse
import tqdm
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from annotate import Annotate
from dataloader import VOXDataset


def main(args):

    vox_dataset = VOXDataset(args)
    print(f"Loaded {len(vox_dataset)} files")
    annotate = Annotate(args)

    all_annotations = []
    # for i, data in enumerate(vox_dataloader):
    for i in tqdm.tqdm(range(0, len(vox_dataset))):
        data = vox_dataset[i]
        audio_file, label_info = data['audio_path'], data['label_info']
        annotations = annotate(audio_file)
        annotations['label_info'] = label_info
        all_annotations.append(annotations)
        
        #save each annotation 
        with open(f'{args.output_path}/annotations.jsonl', 'w') as f:
            for annotation in all_annotations:
                json.dump(annotation, f)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOXCeleb1 Annotator")
    parser.add_argument("--root_path", type=str, help="Root Path of VOX Celeb1 Dataset", required=True)
    parser.add_argument("--vox_metadata_path", type=str, help="Path to vox1_vox1_meta.csv file")
    parser.add_argument("--output_path", type=str, help="Directory to save the output annotation file", required=True)
    parser.add_argument(
        "--asr_model", 
        type=str, 
        help="A Whisper variant supported by Faster-Whisper - https://github.com/SYSTRAN/faster-whisper/",
        default='large-v2'
    )
    parser.add_argument("--device", type=str, help="Device to run the pipeline on (cpu or cuda)", default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--compute_type", type=str, help="Precision to run whisper", default='int8', choices=['int8', 'float16', 'float32'])
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Number of audio splits. Note that this is not the same as the number of audio files that are processed at once.",
        default=1
    )
    parser.add_argument(
        "--fast_transcript", 
        action='store_true', 
        help="Enabling this will not save probabilities and other information of transcription",
    )
    parser.add_argument(
        "--emotion_model", 
        type=str, 
        help="HF emotion model",
        default="bhadresh-savani/bert-base-uncased-emotion",
    )
    parser.add_argument(
        "--disable_asr",
        action='store_true',
        help="Disable ASR pipeline"
    )
    parser.add_argument(
        "--disable_alignment",
        action='store_true',
        help="Disable alignment pipeline"
    )
    parser.add_argument(
        "--disable_emotion",
        action='store_true',
        help="Disable alignment pipeline"
    )


    args = parser.parse_args()
    main(args)


