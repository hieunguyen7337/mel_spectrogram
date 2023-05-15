import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_file = "test.wav"
y, sr = librosa.load(audio_file, sr=16000, mono=True)

# Compute STFT
D = np.abs(librosa.stft(y))

# Convert spectrogram to decibels
D_db = librosa.power_to_db(D, ref=np.max)

# Visualize spectrogram before convert to dB
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.title('spectrogram')
plt.tight_layout()
plt.show()

# Visualize spectrogram after converted
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.title('spectrogram to dB')
plt.tight_layout()
plt.show()