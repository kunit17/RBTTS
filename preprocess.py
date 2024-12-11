
import re
import utils
import librosa
import librosa.display
import numpy as np
import torch
import os
import torch.nn.functional as F


n_fft, hop_length, sr = utils.get_audio_params()

def read_srt_file():
    file_path = utils.get_srt()
    if not file_path:
        raise FileNotFoundError("SRT file path not found in the configuration.")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

input_file = read_srt_file()

# Regular expression
pattern = r'(\d{3}|\d{4})\s+(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=</font>|</i></font>)'

# Find all matches
matches = re.findall(pattern, input_file, re.DOTALL)

data_dict = {match[0]: {"Timestamp": match[1], "Text": match[2]} for match in matches}

vocab = []

#cleaning transcription by moving values from SRT file
for key, value in data_dict.items():
    value['Text'] = (
        value['Text']
        .replace('\n', ' ')
        .replace('/', '')
        .replace('<b><font color="#ffff00">', '')
        .replace('<b><font color="#ffff00"><i>', '')
        .replace('</i></font></b>', '')
        .replace('<i>','')
        .lower()
    )
    value['Text'] = [
        '<SPACE>' if token.isspace() else token # convert spaces into <SPACE>
        for token in re.findall(r'\b\w+\'?\w*|[.,!?]|\s', value['Text']) #convert text into list with words, <SPACE>, and .,!?
    ]
    vocab += value['Text']
    value['Timestamp'] = [utils.t2ms(split) for split in value['Timestamp'].split('-->')]

print(len(data_dict['359']['Text']))
# Assuming data_dict is your dictionary
max_text_key = max(data_dict, key=lambda k: len(data_dict[k]['Text']))
max_text = data_dict[max_text_key]['Text']

print(f"Key with max Text: {max_text_key}. Length in tokens: {len(max_text)} tokens: {max_text}")

#get each unique value of the dictionary
vocab_sorted = sorted(set(vocab))
vocab_sorted.insert(0,'<PAD>')
print(len(vocab_sorted))
print(len(data_dict))
# with open('vocab.json', 'w') as json_file:
#     json.dump(vocab_sorted, json_file)

print(data_dict[max_text_key]['Timestamp'])

# Directory containing .wav files
input_directory = utils.get_audio_samples()
y = {}

# Iterate through all .wav files in the directory
for file_name in os.listdir(input_directory):
    if file_name.endswith(".wav"):  # Process only .wav files
        file_path = os.path.join(input_directory, file_name)
        
        # Load the audio file and normalize amplitude
        signal, _ = librosa.load(file_path, sr=sr)  # Signal normalized between -1 and 1
        
        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        
        # Convert to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        file = os.path.splitext(file_name)[0]
        # Store the log-mel spectrogram as a PyTorch tensor
        y[file] = torch.tensor(log_mel_spectrogram, dtype=torch.float32)

# Example: Print the keys and shapes of tensors in the dictionary
print(f"shape of largest tensor is {y['1581'].shape}")

largest_tensor_key = None
largest_tensor_size = 0

for key, tensor in y.items():
    tensor_size = tensor.numel()  # Get the total number of elements in the tensor
    if tensor_size > largest_tensor_size:
        largest_tensor_size = tensor_size
        largest_tensor_key = key

print(f"Largest tensor is associated with key: {largest_tensor_key}")
print(f"Size of the largest tensor: {largest_tensor_size}")

max_frames = 302

# Pad using -80 dB for silence
for key, mel in y.items():
    pad_amount = max_frames - mel.size(1)  # Calculate padding for time dimension
    y[key] = F.pad(mel, (0, pad_amount), value=-80)  # Update in place

