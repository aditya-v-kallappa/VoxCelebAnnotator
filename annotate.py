import math
import torch
import whisperx
from faster_whisper import WhisperModel

class Annotate:

    def __init__(self, args):
        self.args = args
        if not self.args.disable_asr:
            if self.args.fast_transcript:
                self.asr_model = whisperx.load_model(self.args.asr_model, device=self.args.device, compute_type=self.args.compute_type)
            else:
                self.asr_model = WhisperModel(self.args.asr_model, device=self.args.device, compute_type=self.args.compute_type)
        
        if not self.args.disable_alignment:
            self.model_a, self.metadata = whisperx.load_align_model(language_code=result['language'], device=self.args.device)


    def get_audio_details(self, audio_file):
        samples, sample_rate = sf.read(f'{audio_file}')
        duration = len(samples) / sample_rate

        return {'sample_rate (Hz)'; sample_rate, 'length (s)': duration}

    def get_transcript(self,  audio_file):

        if self.args.fast_transcript:
            audio = whisperx.load_audio(audio_file)
            result = self.asr_model.transcribe(audio, batch_size=self.args.batch_size)

            transcript_info = None
            transcript_result_info = None
            return result, transcript_info, transcript_result_info
        
        else:
            
            segments, transcript_info = model.transcribe(audio_file)
            transcript_result_info = {}
            text = ''
            for i, segment in segments:
                if i == 0:
                    start = segment.start
                text += segment.text
                end = segment.end
            
                transcript_result_info['prob'] = math.exp(segment.avg_logprob)
                transcript_result_info['no_speech_prob'] = segment.no_speech_prob
                transcript_result_info['compression_ratio'] = segment.compression_ratio
            
            result = {
                'segments': [{
                    'start': start,
                    'end': end,
                    'text': text
                }],
                'language': transcript_info.language
            }

            return result, transcript_info, transcript_result_info        

    def align(self, audio_file, segments):  

        # Uses Wav2Vec2 model to align audio with text
        audio = whisperx.load_audio(audio_file)
        result = whisperx.align(segments, self.model_a, self.metadata, audio, self.args.device, return_char_alignments=False)
        diarize_segments = diarize_model(audio)

        #We could set max_speakers = 1 as vox_celeb dataset has audio of only 1 person. But this is something to be tested
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        
        #Assign the speakers to audio segments
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result

    def detect_emotion(self, text):
        classifier = pipeline('text-classification', model=self.args.emotion_model, return_all_scores=True)
        result = classifier(text)
        
        return result

    def annotate_audio(self, audio_file):
        annotations = {}
        annotations['audio_details'] = self.get_audio_details(audio_file)
        if not self.args.disable_asr:
            result, transcript_info, transcript_result_info = self.get_transcript(audio)
            annotations['transcript'] = ' '.join([r['text'] for r in result['segment']])
            annotations['transcript_info'] = transcript_info
            annotations['transcript_result_info'] = transcript_result_info
        
            if not self.args.disable_alignment:
                result = self.align(audio_file, result['segments'])
                annotations['alignment_results'] = result
            else:
                annotations['alignment_results'] = None

            if not self.args.disable_emotion:
                result = self.detect_emotion(annotations['transcript'])
                annotations['emotion'] = result
            else:
                annotations['emotion'] = None
        
        return annotations

    
    def __call__(self, audio_file):
        return self.annotate_audio(audio_file)
            
            


        



        

        
        



