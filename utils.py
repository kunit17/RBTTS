import librosa
import librosa.display
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
from pydub import AudioSegment
import config
from config import CONFIG
from pydub import AudioSegment
import os
import torch



def get_train_params():
    if 'train_params' not in CONFIG:
        raise KeyError("'train_params' key not found in CONFIG.")
    batch_size = CONFIG['train_params']['batch_size']
    learning_rate = CONFIG['train_params']['learning_rate']
    epochs = CONFIG['train_params']['epochs']
    block_size = CONFIG['train_params']['block_size']
    char_size = CONFIG['train_params']['char_size']
    d_model = CONFIG['train_params']['d_model']
    n_heads = CONFIG['train_params']['n_heads']
    dropout_rate = CONFIG['train_params']['dropout_rate']
    head_size = CONFIG['train_params']['head_size']
    return batch_size, learning_rate, epochs, block_size, char_size, d_model, n_heads, dropout_rate, head_size

def get_audio_params():
    n_fft = CONFIG['audio_params']['n_fft']
    hop_length = CONFIG['audio_params']['hop_length']
    sr = CONFIG['audio_params']['sr']
    return n_fft, hop_length, sr

def get_audio_samples():
    return config.DATA_PATHS.get('training_samples', None)

def get_training_data():
    return config.DATA_PATHS.get('training_data', None)

def get_srt():
    return config.DATA_PATHS.get('srt_file_path', None)

def get_whole_audio():
    return config.DATA_PATHS.get('whole_audio_path', None)


def t2ms(time_str):
    # Split the time into the part before and after the comma
    time, milliseconds = time_str.split(',')
    
    # Split the time into hours, minutes, and seconds
    hours, minutes, seconds = map(int, time.split(':'))
    
    # Convert everything to milliseconds
    total_ms = (hours * 3600000) + (minutes * 60000) + (seconds * 1000) + int(milliseconds)
    return total_ms


def export_audio_segments(whole_audio_path, data_dict, output_directory="./Data/Training_Samples/"):
    """
    Extracts audio segments based on timestamps in a data dictionary
    and exports them as separate .wav files.
    
    Args:
        whole_audio_path (str): Path to the whole audio file.
        data_dict (dict): Dictionary where keys are identifiers and values contain timestamps.
                          Example: {1: {"Timestamp": [start_ms, end_ms]}, ...}
        output_directory (str): Directory to save the exported audio segments.
                                Defaults to "./Data/Training_Samples/".
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Load the whole audio file
    audio = AudioSegment.from_wav(whole_audio_path)
    
    # Iterate through the dictionary and process each segment
    for key, value in data_dict.items():
        # Extract start and end times in milliseconds
        start_t, end_t = value["Timestamp"]
        
        # Slice the audio
        cut_audio = audio[start_t:end_t]
        
        # Define the output file path
        output_file = os.path.join(output_directory, f"{key}.wav")
        
        # Export the audio segment
        cut_audio.export(output_file, format="wav")
        
        print(f"Exported: {output_file}")

class Tokenizer:
    def __init__(self, chars):
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        self.chars = chars
        self.word_to_idx = {word: i for i, word in enumerate(chars)}
        self.idx_to_word = {i: word for i, word in enumerate(chars)}

    def split_into_chars(self, text):
        # Split text into individual characters with <SPACE> between words
        result = []
        special_tokens = ['<SPACE>', '!', ',', '.', '?']
        for word in text:
            if word in special_tokens:
                result.append(word)
            else:
                for char in word:
                    result.append(char)  # Add each character individually
        return result

    def encode(self, text):

        encoded_sentence = []
        for word in text:
            print(word)
            if word in self.word_to_idx:
                encoded_sentence.append(self.word_to_idx[word])  # Map known words directly
            else:
                # Split unknown words into characters
                encoded_sentence += [self.word_to_idx.get(char, self.word_to_idx[self.unk_token]) for char in word]

        # Append EOS token
        encoded_sentence.append(self.word_to_idx[self.eos_token])
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.long)
        return encoded_sentence



