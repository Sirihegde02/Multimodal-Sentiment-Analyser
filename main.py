from processors.visual_processor import VisualProcessor
from processors.audio_processor import AudioProcessor
import cv2
import numpy as np
import sounddevice as sd
import time

class MultimodalSentimentAnalyzer:
    def __init__(self):
        print("Initializing Multimodal Sentiment Analyzer...")
        self.visual_processor = VisualProcessor()
        self.audio_processor = AudioProcessor()
        self.audio_buffer = []
        self.sample_rate = 16000
        print("Initialization complete!")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio processing"""
        if status:
            print(f"Audio callback status: {status}")
        self.audio_buffer.extend(indata[:, 0])  # Store first channel

    def analyze_audio(self):
        """Analyze the audio buffer"""
        if len(self.audio_buffer) > self.sample_rate:  # Process 1 second of audio
            audio_chunk = np.array(self.audio_buffer[-self.sample_rate:])
            audio_features = self.audio_processor.process_audio(audio_chunk)
            self.audio_buffer = self.audio_buffer[-self.sample_rate:]  # Keep last second
            return audio_features
        return None

    def analyze_frame(self, frame):
        """Analyze a single video frame"""
        return self.visual_processor.process_frame(frame)

    def get_combined_sentiment(self, visual_features, audio_features):
        """Combine visual and audio features to determine sentiment"""
        sentiments = ['negative', 'neutral', 'positive']
        visual_weight, audio_weight = 0.6, 0.4
        scores = np.zeros(3)

        # Combine visual sentiment
        if visual_features and 'emotion_probs' in visual_features:
            # Flatten or average emotion probabilities
            visual_probs = np.mean(visual_features['emotion_probs'], axis=0)
            
            # Ensure visual_probs is a 1D array
            if visual_probs.ndim > 1:
                visual_probs = visual_probs.flatten()
            
            # Map 7 emotions to 3 sentiment categories
            emotion_to_sentiment = {
                'angry': 0, 'disgust': 0, 'fear': 0,  # Negative
                'happy': 2, 'surprise': 2,            # Positive
                'sad': 0, 'neutral': 1               # Neutral
            }
            sentiment_probs = np.zeros(3)
            for i, emotion in enumerate(self.visual_processor.emotions):
                sentiment_idx = emotion_to_sentiment.get(emotion, 1)  # Default to neutral
                sentiment_probs[sentiment_idx] += visual_probs[i]
            scores += visual_weight * sentiment_probs

        # Combine audio sentiment
        if audio_features and 'sentiment' in audio_features:
            sentiment = audio_features['sentiment']
            sentiment_idx = sentiments.index(sentiment['label'])
            scores[sentiment_idx] += audio_weight * sentiment['score']

        # Determine final sentiment
        if np.sum(scores) > 0:
            normalized_scores = scores / np.sum(scores)
            sentiment_idx = np.argmax(normalized_scores)
            return sentiments[sentiment_idx], normalized_scores[sentiment_idx]
        return 'neutral', 0.5



    def run(self):
        """Run the real-time sentiment analysis"""
        print("Starting real-time sentiment analysis. Press 'q' to exit.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return

        # Start audio stream
        audio_stream = sd.InputStream(channels=1, samplerate=self.sample_rate, callback=self.audio_callback)
        audio_stream.start()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame.")
                    break

                visual_features = self.analyze_frame(frame)
                audio_features = self.analyze_audio()
                sentiment, confidence = self.get_combined_sentiment(visual_features, audio_features)

                # Display results
                display_text = f"Sentiment: {sentiment} ({confidence:.2f})"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow('Multimodal Sentiment Analysis', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            audio_stream.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = MultimodalSentimentAnalyzer()
    analyzer.run()
