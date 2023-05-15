import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_file = "test.wav"
y, sr = librosa.load(audio_file, sr=16000, mono=True)

# Convert STFT to Mel scale
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert Mel spectrogram to decibels
S_db = librosa.power_to_db(S, ref=np.max)

# Visualize mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.title('Mel spectrogram to dB')
plt.tight_layout()
plt.show()

# Save mel spectrogram as NPY file
np.save('mel_spectrogram.npy', S_db)