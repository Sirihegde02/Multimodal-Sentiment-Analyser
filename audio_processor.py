import numpy as np
import torch
import torch.nn as nn
import librosa
import os

class AudioEmotionCNN(nn.Module):
    def __init__(self):
        super(AudioEmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * 16, 512)
        self.fc2 = nn.Linear(512, 7)  # 7 emotions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AudioProcessor:
    def __init__(self):
        print("Initializing Audio Processor...")
        self.sample_rate = 16000
        self.emotion_model = AudioEmotionCNN()
        
        # Load pre-trained weights
        try:
            weights_path = 'models/audio_emotion_model.pth'
            if os.path.exists(weights_path):
                self.emotion_model.load_state_dict(torch.load(weights_path))
                print("Loaded audio emotion model weights")
            else:
                print("Warning: Audio emotion model weights not found")
        except Exception as e:
            print(f"Error loading audio model weights: {e}")
            
        self.emotion_model.eval()
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def extract_features(self, audio):
        try:
            if audio is None or len(audio) == 0:
                return None

            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=64)
            
            # Ensure correct dimensions
            if mfccs.shape[1] < 64:
                pad_width = ((0, 0), (0, 64 - mfccs.shape[1]))
                mfccs = np.pad(mfccs, pad_width, mode='constant')
            else:
                mfccs = mfccs[:, :64]
            
            # Reshape for model (batch_size, channels, time)
            features_tensor = torch.FloatTensor(mfccs[0:1]).unsqueeze(0)  # Take first MFCC
            
            # Get emotion predictions
            with torch.no_grad():
                emotion_outputs = self.emotion_model(features_tensor)
                emotion_probs = torch.softmax(emotion_outputs, dim=1)
            
            # Get predicted emotion
            emotion_idx = emotion_probs.argmax().item()
            emotion = self.emotions[emotion_idx]
            confidence = emotion_probs[0][emotion_idx].item()
            
            # Map emotion to sentiment
            sentiment = self.emotion_to_sentiment(emotion, confidence)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'sentiment': sentiment,
                'features': mfccs
            }

        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None

    def emotion_to_sentiment(self, emotion, confidence):
        """Map emotion to sentiment with confidence"""
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        
        if emotion in positive_emotions:
            return {'label': 'positive', 'score': confidence}
        elif emotion in negative_emotions:
            return {'label': 'negative', 'score': confidence}
        else:
            return {'label': 'neutral', 'score': confidence}

    def process_audio(self, audio_chunk):
        return self.extract_features(audio_chunk)