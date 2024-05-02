import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class VOXDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_path = args.root_path
        speaker_metadata_path = args.vox_metadata_path if args.vox_metadata_path else f"{self.root_path}/vox1_vox1_meta.csv"
        self.speaker_metadata = pd.read_csv(f"{speaker_metadata_path}", sep='\t')

        self.wav_files = glob.glob(f"{self.root_path}/wav/**/*.wav", recursive=True)
        self.txt_files = [wav.replace("/wav/", "/txt/").replace(".wav", ".txt") for wav in self.wav_files]


    def __len__(self):
        return len(self.wav_files)

    def extract_metadata(self, speaker_id):
        return self.speaker_metadata[self.speaker_metadata['VoxCeleb1 ID'] == speaker_id]

    def extract_info_from_text_file(self, txt_file_path):
        # Relevant Info to be extracted
        label_info = {
            'speaker_id': None,
            'yt_link': None,
            'offset': None,
            'fv_conf': None,
            'asd_conf': None,
            'speaker': None,
            'frame_start': None,
            'frame_end': None
        }
        frames = []

        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

            # Loop through each line
            for line in lines:
                if line.startswith('Identity'):
                    label_info['speaker_id'] = line.split(':')[1].strip()
                elif line.startswith('Reference'):
                    label_info['yt_link'] = line.split(':')[1].strip()
                elif line.startswith('Offset'):
                    label_info['offset'] = line.split(':')[1].strip()
                elif line.startswith('FV Conf'):
                    label_info['fv_conf'] = line.split(':')[1].strip()
                elif line.startswith('ASD Conf'):
                    label_info['asd_conf'] = line.split(':')[1].strip()
                elif line.startswith('FRAME'):
                    continue
                elif line.strip():
                    frames.append(line.split())

        # Get frame end
        label_info['metadata'] = self.extract_metadata(label_info['speaker_id']).to_dict(orient='records')
        label_info['frame_start'] = int(frames[0][0])
        label_info['frame_end'] = int(frames[-1][0])

        return label_info

    def __getitem__(self, index):

        audio = self.wav_files[index]
        label_info = self.extract_info_from_text_file(self.txt_files[index])
        return {'audio_path': audio, 'label_info': label_info}

