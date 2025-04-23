import cv2
import numpy as np
from datetime import datetime

def load_emotion_detector():
    # Load improved cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    return face_cascade, eye_cascade

def enhance_image(frame):
    # Enhanced image preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def detect_facial_features(face_roi, eye_cascade):
    # Detect eyes to validate face detection
    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return len(eyes) >= 2

def analyze_expression(face_roi):
    # Simple expression analysis based on pixel intensity and gradients
    try:
        # Calculate histogram features
        hist = cv2.calcHist([face_roi], [0], None, [256], [0,256])
        hist_norm = hist.flatten() / sum(hist.flatten())
        
        # Calculate gradient features
        sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Analyze features
        avg_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        gradient_mean = np.mean(magnitude)
        
        # Simple rule-based classification
        if gradient_mean > 20 and std_intensity > 45:
            if avg_intensity > 130:
                return 'Happy', 0.7
            elif avg_intensity < 110:
                return 'Sad', 0.6
            else:
                return 'Surprised', 0.65
        elif gradient_mean < 15:
            return 'Neutral', 0.8
        elif std_intensity > 50:
            return 'Angry', 0.6
        else:
            return 'Neutral', 0.5
            
    except Exception as e:
        print(f"Error in expression analysis: {e}")
        return 'Unknown', 0.0

def apply_smoothing(predictions, window_size=5):
    if len(predictions) < window_size:
        return predictions[-1] if predictions else ('Unknown', 0.0)
    
    recent_preds = predictions[-window_size:]
    emotion_counts = {}
    conf_sums = {}
    
    for emotion, conf in recent_preds:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        conf_sums[emotion] = conf_sums.get(emotion, 0) + conf
    
    max_count_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    avg_conf = conf_sums[max_count_emotion] / emotion_counts[max_count_emotion]
    
    return max_count_emotion, avg_conf

def detect_emotions():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_detector, eye_cascade = load_emotion_detector()
    prediction_history = []
    
    prev_time = datetime.now()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        current_time = datetime.now()
        time_diff = (current_time - prev_time).total_seconds()
        if time_diff > 0:
            fps = 1 / time_diff
        prev_time = current_time
        
        # Enhanced preprocessing
        enhanced = enhance_image(frame)
        
        # Detect faces with optimized parameters
        faces = face_detector.detectMultiScale(
            enhanced,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            # Add padding
            padding = int(0.1 * w)
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, frame.shape[1])
            y2 = min(y + h + padding, frame.shape[0])
            
            face_roi = enhanced[y1:y2, x1:x2]
            
            try:
                # Validate face using eye detection
                if not detect_facial_features(face_roi, eye_cascade):
                    continue
                
                # Analyze expression
                emotion, confidence = analyze_expression(face_roi)
                
                # Add to prediction history
                prediction_history.append((emotion, confidence))
                if len(prediction_history) > 10:
                    prediction_history.pop(0)
                
                # Get smoothed prediction
                smooth_emotion, smooth_conf = apply_smoothing(prediction_history)
                
                # Draw face rectangle with dynamic color based on confidence
                color = (0, int(255 * smooth_conf), 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display emotion and confidence
                text = f"{smooth_emotion}: {smooth_conf:.2f}"
                cv2.putText(frame, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Enhanced Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Enhanced Emotion Detection...")
    print("Press 'q' to quit")
    detect_emotions()
