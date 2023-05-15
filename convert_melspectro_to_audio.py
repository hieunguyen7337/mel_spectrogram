import librosa
import librosa.display
import soundfile as sf
import numpy as np

# Load mel spectrogram
mel_spectrogram = np.load('mel_spectrogram.npy')

# Revert mel spectrogram db normalization
S_db_to_power = librosa.db_to_power(mel_spectrogram, ref=3190.1357)

# Convert mel spectrogram back to linear spectrogram
audio_array = librosa.feature.inverse.mel_to_audio(S_db_to_power, sr=16000)

# Save waveform as WAV file
sf.write('audio.wav', audio_array, 16000)