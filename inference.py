librosa.display.waveshow(y=signal, sr=sr)
plt.title('Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()

plt.figure(figsize=(10, 4))
img = librosa.display.specshow(
    log_mel_spectrogram, 
    sr=sr, 
    hop_length=hop_length, 
    x_axis='time', 
    y_axis='mel', 
    fmax=8000, 
    cmap='magma'
)
plt.colorbar(img, label='DB')  # Add a label to the colorbar
plt.title('Log Mel Spectrogram')
plt.xlabel('Time (s)')  # Label for the x-axis
plt.ylabel('Frequency (Hz)')  # Label for the y-axis
plt.grid(True, linestyle='--', alpha=0.6)  # Optional: add grid for better readability
plt.tight_layout()
plt.show()



# Reverse Process
mel_spectrogram_reversed = librosa.db_to_power(log_mel_spectrogram, ref=np.max(mel_spectrogram))
linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram_reversed, sr=sr, n_fft=n_fft)

# Use Griffin-Lim to reconstruct the signal
reconstructed_signal = librosa.griffinlim(linear_spectrogram, hop_length=hop_length, n_iter=64)

# Save reconstructed audio
sf.write('reconstructed_audio.wav', reconstructed_signal, sr)



output_file = 'output.wav'  # Specify the output file name

# Ensure the signal is scaled to the appropriate range for WAV files
# WAV files typically expect 16-bit PCM, so we scale the signal to int16
scaled_signal = (signal * 32767).astype(np.int16)

# Write the WAV file
write(output_file, sr, scaled_signal)

print(f"WAV file written to {output_file}")