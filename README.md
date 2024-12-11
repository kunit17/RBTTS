# RBTTS

Embarking on my second project, albeit one that could prove to be too challenging, as it delves into uncharted territory for me and demands skills I’m still mastering. My goal is to produce a natural-sounding, Text-To-Speech model using the Transformer architecture. Learning by coding directly (compared to online courses) has accelerated my knowledge base, however, so at the minimum I expect to gain a lot of insight into Transformers, in general.

# Steps

## Pre-processing:
1) Gather suitable audio files and transcriptions (the normal route is to use publicly available data - I'll be using something more fun, however)
2) Create a tokenizer and encoder that uses padding as the input text I will be training on is variable in length. The choice of tokenizer is very important; I will begin with words, but will eventually use phonemes.
3) Express audio (wav) files as tensors using the Mel Scale
4) Ensure the tensors produced are correctly padded in such a way that the padding does not affect loss calculations

## Model Coding:
1) TBC

## Post-processing:
1) [Vocoder](#vocoder)


![Text-to-speech](./Files/NN_wav.jpg)


# Intuition of a Text-to-Speech Model

## Physics of Sound Waves and Representing them as tensors

When sound is converted digitally onto a computer, the amplitude of the sound wave is sampled at a specified timestep. For example, if sound is being recorded at 22.05 kHz (a common fidelity for recorded sounds), each second the value of the sound wave's amplitude is recorded 22,050 times! Typically, these audio files (or WAV files) are stored on a computer where each timestep's amplitude is represented by 16-bits of a computer's memory. What this means is that, for each time step, the signal coming from the audio can be represented by a number bewteen -32,767 to 32,737 - and this happens 22,050 separate times in each second - that's a lot of granularity!

### Step 1: Convert WAV file into numbers representing the sound wave's amplitude at each timestep (1D NP.array)
Purpose: Representing an audio wave as numbers allows us to perform calculations and manipulate the numbers
![Raw Wav](./Files/Raw_Wav.png)

### Step 2: Short-Time Fourier Transform (STFT)
Purpose: Converts the time-domain signal into a time-frequency representation.

### 3Blue1Brown YouTube Explanation
[![Watch on YouTube](https://img.youtube.com/vi/spUNpyF58BY/0.jpg)](https://www.youtube.com/watch?v=spUNpyF58BY&t=31s)

### Step 3: Magnitude Spectrogram
Purpose: Computes the magnitude (absolute value) of the complex STFT values, discarding phase information

### Step 4: Mel Spectrogram
Purpose: Projects the spectrogram onto a Mel scale (perceptual frequency scale).
n_mels: Number of Mel bands (e.g., 80 for speech processing).
Mel scale focuses on lower frequencies, which are more relevant for human perception.

### Step 5: Log-Mel Spectrogram:
Purpose: Converts the Mel spectrogram into a logarithmic scale (similar to decibels).
Logarithmic scaling is essential because human hearing perceives sound intensity logarithmically.
![Mel Spectrograpm](./Files/Mel_Spec.png)

### Further resources
[Understanding sound waves and their numerical representations](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)


# Vocoder
After we perform calculations on each sample, we would like to have a way to reassmble the numbers back into audio waves. However, it is not as simple as just reversing the math equations we applied to the initial numbers. One issue is that when the audio wave was initially converted into tensors, information is lost during this process (specifically when projecting the magnitude spectrogram onto a Mel scale - [Step 4](#step-4-mel-spectrogram))
Example of what it sounds like without using a vocoder: [no_vocoder_output.wav](no_vocoder_output.wav)


## Files explained

1) config.json -> contain the model's hyperparameters
2) txtpreprocessing -> takes in sample text followed by cleaning, tokenizing, and encoding for the model's inputs (x)
3) voicepreprocessing -> takes in sample sound files (wavs), does maths on them to convert them into tensors for the model's targets (y)
