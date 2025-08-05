import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time
from collections import Counter

# --- Configuration ---
MODEL_PATH = "src/model/fast_emotion_model.h5"
model = load_model(MODEL_PATH)


def preprocess_face_fast(face):
    """Preprocesses face for the custom CNN model."""
    if face.size == 0:
        return None
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_normalized = face_resized.astype(np.float32) / 255.0
    face_batch = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=-1)
    return face_batch

# --- Frame Processing Settings ---
def get_emotion():
    offset = 10
    MAX_FRAMES = 40
    FRAME_RATE = 0.25
    last_frame_time = 0
    frame_count = 0
    emotion_list = []
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # --- Main Execution ---
    cap = cv2.VideoCapture(0)
    print("🎥 Starting emotion detection... Press 'q' to quit.")
    print("ℹ  Look at the console for detection status messages.")

    # ✨ CORRECTION: Lowered confidence for more reliable detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as face_detection:
        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            
            if (current_time - last_frame_time) >= FRAME_RATE:
                last_frame_time = current_time
                results = face_detection.process(image_rgb)
                
                # The entire labeling logic is inside this 'if' block.
                if results.detections:
                    # ✨ DEBUG: This message will appear if a face is found.
                    print("✅ Face Detected!")
                    
                    for detection in results.detections:

                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1, y1 = max(0, int(bbox.xmin * w)), max(0, int(bbox.ymin * h))
                        x2, y2 = min(w, int((bbox.xmin + bbox.width) * w)), min(h, int((bbox.ymin + bbox.height) * h))
                        
                        face = frame[y1-offset:y2+offset, x1-offset:x2+offset]
                        face_processed = preprocess_face_fast(face)
                        
                        if face_processed is not None:
                            prediction = model.predict(face_processed, verbose=0)
                            emotion = emotion_labels[np.argmax(prediction)]
                            
                            emotion_list.append(emotion)
                            frame_count += 1

                            display_text = emotion.upper()
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.8
                            thickness = 3
                            
                            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                            text_x = (w - text_width) // 2
                            text_y = text_height + 25

                            overlay = frame.copy()
                            cv2.rectangle(overlay, (text_x - 10, text_y - text_height - 15), (text_x + text_width + 10, text_y + 15), (0,0,0), -1)
                            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                            
                            
                else:
                    # ✨ DEBUG: This message appears if no face is found.
                    print("❌ No face detected.")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Emotion detection stopped!")

    if emotion_list:
        top_2 = Counter(emotion_list).most_common(2)
        print("\n--- Emotion Summary ---")
        mood_list = []
        for i, (emotion, count) in enumerate(top_2):
            mood_list.append(emotion)
    return mood_list