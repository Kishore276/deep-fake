import torch
import timm
import cv2
import numpy as np
import tempfile
import os
import re
import base64
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import PIL.Image as Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load the pretrained model for image/video deepfake detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("tf_efficientnet_b4", pretrained=True, num_classes=1)

# Add a second model with different architecture for ensemble detection
model2 = timm.create_model("resnet50", pretrained=True, num_classes=1)

model.eval().to(device)
model2.eval().to(device)

# Load text detection model
text_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
text_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
text_model.eval().to(device)

# Modified Detection Thresholds to reduce false positives
THRESHOLD_IMAGE = 0.85  # Increased from 0.7
THRESHOLD_VIDEO = 0.8  # Increased from 0.6
FAKE_VIDEO_PERCENTAGE = 60  # Increased from 40

# Face detection for more accurate analysis
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Enhanced image preprocessing function with denoising
def preprocess_frame(frame):
    # Apply denoising to reduce compression artifacts
    denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(denoised_frame).unsqueeze(0).to(device)

# Convert frame to base64 for web display
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Enhanced prediction function using ensemble approach
def predict_frame(frame):
    # Process full frame with both models
    img_tensor = preprocess_frame(frame)
    
    with torch.no_grad():
        output1 = model(img_tensor)
        output2 = model2(img_tensor)
    
    score1 = torch.sigmoid(output1).item()
    score2 = torch.sigmoid(output2).item()
    
    # Use weighted average of models
    # Weight the EfficientNet model less if it's been giving false positives
    ensemble_score = (score1 * 0.3) + (score2 * 0.7)
    
    return ensemble_score

# Face-specific analysis function
def analyze_face_regions(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    
    # If no faces found, use the regular frame analysis but with lower weight
    if len(faces) == 0:
        return predict_frame(frame) * 0.8
        
    # Analyze each face separately
    face_scores = []
    for (x, y, w, h) in faces:
        # Add padding around the face (20% extra)
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        
        # Ensure coordinates are within frame boundaries
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(frame.shape[1], x + w + padding_x)
        y2 = min(frame.shape[0], y + h + padding_y)
        
        face_img = frame[y1:y2, x1:x2]
        
        # Only process if face region is valid
        if face_img.size > 0:
            # Process just the face region
            face_score = predict_frame(face_img)
            face_scores.append(face_score)
    
    # Return max face score or fall back to full frame analysis
    if face_scores:
        return max(face_scores)
    else:
        return predict_frame(frame)

# Process video with frame visualization and improved analysis
def process_video(video_bytes):
    try:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_bytes)
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            os.remove(temp_video.name)
            return {"result": "Error: Empty or corrupt video file!", "frames": []}

        # Use specific frame positions: 50, 100, 150, etc.
        frame_indices = []
        frame_position = 10  # Start with frame 50
        
        while frame_position < total_frames:
            frame_indices.append(frame_position)
            frame_position += 10  # Increment by 50 for next position
            
        fake_count = 0
        processed_frames = []
        confidence_scores = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Rest of your frame processing code...
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use face-specific analysis for more accurate detection
            prediction = analyze_face_regions(rgb_frame)
            confidence_scores.append(prediction)
            
            # Add prediction label to frame
            label = f"Frame {idx}: {'Fake' if prediction > THRESHOLD_VIDEO else 'Real'} ({prediction:.2f})"
            color = (0, 0, 255) if prediction > THRESHOLD_VIDEO else (0, 255, 0)
            
            # Convert back to BGR for OpenCV
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add frame to results
            processed_frames.append({
                "image": frame_to_base64(display_frame),
                "prediction": prediction,
                "is_fake": prediction > THRESHOLD_VIDEO
            })
            
            if prediction > THRESHOLD_VIDEO:
                fake_count += 1

        cap.release()
        os.remove(temp_video.name)  # Delete temp file after processing
        
        # Calculate fake percentage
        fake_percentage = (fake_count / len(processed_frames)) * 100 if processed_frames else 0
        
        # Add temporal consistency check (detect sudden changes between adjacent frames)
        is_consistent = True
        if len(confidence_scores) > 3:
            diffs = [abs(confidence_scores[i] - confidence_scores[i-1]) for i in range(1, len(confidence_scores))]
            avg_diff = sum(diffs) / len(diffs)
            # If average difference is very small, this might be AI-generated with consistent fake patterns
            # If average difference is very large, this might be suspicious due to inconsistency
            is_consistent = 0.05 < avg_diff < 0.2
        
        # Final decision with multiple factors
        is_fake = fake_percentage > FAKE_VIDEO_PERCENTAGE
        
        # Adjust based on temporal consistency
        if not is_consistent and fake_percentage > 30 and fake_percentage < 70:
            # In borderline cases, reduce confidence in fake detection
            is_fake = False
            
        result = "Fake Video (Deepfake)" if is_fake else "Real Video"
        
        return {
            "result": result,
            "frames": processed_frames,
            "fake_percentage": fake_percentage
        }
    
    except Exception as e:
        return {"result": f"Error: {str(e)}", "frames": []}

# Improved fake image detection with better error handling
def detect_fake_image(file_bytes):
    try:
        # Create a fresh BytesIO object from the bytes
        img_io = BytesIO(file_bytes)
        
        # Try opening with PIL
        img = Image.open(img_io).convert("RGB")
        img_np = np.array(img)
        
        # First do face-specific analysis
        rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        prediction = analyze_face_regions(rgb_img)
        
        # Add prediction label to image
        display_img = img_np.copy()
        label = f"{'Fake' if prediction > THRESHOLD_IMAGE else 'Real'} ({prediction:.2f})"
        color = (0, 0, 255) if prediction > THRESHOLD_IMAGE else (0, 255, 0)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        cv2.putText(display_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return {
            "result": "Fake Image (Deepfake)" if prediction > THRESHOLD_IMAGE else "Real Image",
            "prediction": prediction,
            "image": frame_to_base64(display_img)
        }
    except Exception as e:
        # Add more detailed error information for debugging
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)  # Log the error for debugging
        return {"result": error_msg, "image": "", "error_details": "Failed to process image"}

# Detect fake text (unchanged)
def detect_fake_text(text):
    try:
        # Pre-process input text
        # Check for minimum length
        if len(text.strip()) < 50:
            return {
                "result": "Text too short for reliable analysis",
                "confidence": 0.5
            }
            
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            
            # For specialized AI detection models, logits[0] is often human, logits[1] is AI
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=0)
            
            # Get probability of being AI-generated
            ai_prob = probabilities[1].item()
            human_prob = probabilities[0].item()
            
            # Calculate perplexity (another useful metric for detection)
            # Higher perplexity often indicates more natural/human text
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Apply statistical heuristics to improve accuracy
            # For example, extremely regular text patterns often indicate AI generation
            sentence_lengths = [len(s.split()) for s in text.split('.') if s.strip()]
            length_variance = np.var(sentence_lengths) if sentence_lengths else 0
            
            # Adjust confidence based on multiple factors
            confidence = ai_prob
            
            # If variance is very low, boost AI probability (very regular text is suspicious)
            if length_variance < 2.0 and len(sentence_lengths) > 5:
                confidence = min(confidence + 0.1, 0.95)
                
            # Text with extremely predictable patterns might be AI
            repetition_patterns = len(re.findall(r'(\b\w+\b)(?:\s+\w+){1,5}\s+\1\b', text, re.IGNORECASE))
            if repetition_patterns > 3:
                confidence = min(confidence + 0.05, 0.95)
                
            result = "AI-Generated Text" if confidence > 0.7 else "Human-Written Text"
            
            # Add detail for ambiguous cases
            if 0.4 < confidence < 0.7:
                result = "Possibly " + result + " (Uncertain)"
        
        return {
            "result": result,
            "confidence": confidence,
            "human_probability": human_prob,
            "ai_probability": ai_prob
        }
    except Exception as e:
        print(f"Text detection error: {str(e)}")
        return {"result": f"Error: {str(e)}", "confidence": 0}
def detect_fake(file):
    try:
        file_bytes = file.read()
        
        # Try to determine file type using file extension first
        filename = file.filename.lower()
        
        # Check if we can use magic library for more accurate file type detection
        try:
            from magic import Magic
            mime = Magic(mime=True)
            file_type = mime.from_buffer(file_bytes)
            
            if file_type.startswith('image/'):
                return detect_fake_image(file_bytes)
            elif file_type.startswith('video/'):
                return process_video(file_bytes)
            else:
                return {"result": f"Unsupported file type: {file_type}", "frames": []}
        except ImportError:
            # Fall back to extension-based detection if magic is not available
            if any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
                return detect_fake_image(file_bytes)
            elif any(filename.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.wmv']):
                return process_video(file_bytes)
            else:
                # Last resort: try to detect format based on content
                try:
                    img_io = BytesIO(file_bytes)
                    Image.open(img_io)  # Just to check if it's a valid image
                    img_io.seek(0)  # Reset position
                    return detect_fake_image(file_bytes)
                except:
                    # If image detection fails, try as video
                    return process_video(file_bytes)
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)  # Log for debugging
        return {"result": error_msg, "frames": []}

# Flask routes (unchanged)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Handle file upload
            if "file" in request.files:
                file = request.files["file"]
                if file.filename == "":
                    return jsonify({"error": "No file selected!"})
                
                return jsonify(detect_fake(file))
            
            # Handle text analysis
            elif "text" in request.form:
                text = request.form["text"]
                if not text.strip():
                    return jsonify({"error": "No text provided!"})
                
                return jsonify(detect_fake_text(text))
            
            else:
                return jsonify({"error": "No data provided!"})

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(error_msg)  # Log for debugging
            return jsonify({"error": error_msg})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)