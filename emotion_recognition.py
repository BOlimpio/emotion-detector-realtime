from pathlib import Path
import cv2
import torch
from torchvision import transforms

# Load the TorchScript model for inference (no pickle issues)
ts_model_path = Path('emotion_detector_ts.pt')
assert ts_model_path.exists(), f"TorchScript model not found: {ts_model_path}"
scripted_model = torch.jit.load(str(ts_model_path)).eval()

# Define preprocessing pipeline matching training transforms
def preprocess_face(face_img):
    """
    Convert a NumPy BGR face crop to a normalized tensor for model input,
    with histogram equalization and smoothing.
    """
    # Convert to grayscale
    gray_crop = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray_crop)
    # Apply slight Gaussian blur to reduce noise
    eq = cv2.GaussianBlur(eq, (3,3), 0)
    # Convert back to RGB 3-channel PIL image
    from PIL import Image
    pil = Image.fromarray(eq).convert('RGB')
    # Define transforms: resize to 224, to tensor, normalize
    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tfms(pil)

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("Starting real-time emotion detection. Press 'q' to quit.")

# Determine device and move model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scripted_model.to(device)

# Label mapping (must match training order)
# You can check in Kaggle notebook in line print("Label order:", list(dls.vocab)) 
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# State for temporal smoothing: require stable prediction for threshold frames before update
threshold = 5
last_pred_idx = None
stable_count = 0
# Initial display index
display_idx = None

# Real-time inference loop with bounding-box and probability smoothing
# Buffers for smoothing face box and probabilities
prob_buffer = []
box_buffer = []
buffer_size = 5

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # stricter face detection parameters for stability
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=6, minSize=(100,100)
    )
    if len(faces)>0:
        # select largest face by area
        areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in faces]
        _, (x,y,w,h) = max(areas, key=lambda t: t[0])
        # smooth bounding box
        box_buffer.append((x,y,w,h))
        if len(box_buffer)>buffer_size: box_buffer.pop(0)
        # average box coordinates
        xs, ys, ws, hs = zip(*box_buffer)
        ax, ay, aw, ah = int(sum(xs)/len(xs)), int(sum(ys)/len(ys)), int(sum(ws)/len(ws)), int(sum(hs)/len(hs))
        face_crop = frame[ay:ay+ah, ax:ax+aw]
        tensor = preprocess_face(face_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = scripted_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        # update probability buffer
        prob_buffer.append(probs)
        if len(prob_buffer)>buffer_size: prob_buffer.pop(0)
        # compute averaged probabilities
        avg_probs = sum(prob_buffer)/len(prob_buffer)
        idx = int(avg_probs.argmax())
        label_str = f"{labels[idx]} ({avg_probs[idx]*100:.1f}%)"
        # draw stabilized rectangle and label
        cv2.rectangle(frame, (ax,ay), (ax+aw, ay+ah), (0,255,0), 2)
        cv2.putText(frame, label_str, (ax, ay-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2, cv2.LINE_AA)
    # display frame
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
