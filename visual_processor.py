import cv2
import numpy as np
from mtcnn import MTCNN
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 7)  # 7 emotions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisualProcessor:
    def __init__(self):
        print("Initializing Visual Processor...")
        self.face_detector = MTCNN()
        self.emotion_model = EmotionCNN()
        self.emotion_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def extract_features(self, frame):
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.face_detector.detect_faces(rgb_frame)
            
            if not faces:
                return None

            features = {
                'emotion_probs': [],
                'boxes': [],
                'emotions': []
            }

            for face in faces:
                # Get face box
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                
                # Extract face region
                face_img = rgb_frame[y:y+h, x:x+w]
                
                # Convert to PIL Image
                pil_image = Image.fromarray(face_img)
                
                # Transform image
                face_tensor = self.transform(pil_image).unsqueeze(0)
                
                # Get emotion predictions
                with torch.no_grad():
                    emotion_outputs = self.emotion_model(face_tensor)
                    emotion_probs = torch.softmax(emotion_outputs, dim=1)
                
                # Get predicted emotion
                emotion_idx = emotion_probs.argmax().item()
                emotion = self.emotions[emotion_idx]
                confidence = emotion_probs[0][emotion_idx].item()
                
                # Map emotion to sentiment
                sentiment = self.emotion_to_sentiment(emotion, confidence)
                
                features['emotion_probs'].append(emotion_probs.numpy())
                features['boxes'].append(face['box'])
                features['emotions'].append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'sentiment': sentiment
                })
                
                # Draw on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)

            return features

        except Exception as e:
            print(f"Error in extract_features: {e}")
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

    def process_frame(self, frame):
        if frame is None:
            return None
            
        return self.extract_features(frame)