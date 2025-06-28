import cv2
import numpy as np
import time
import threading
from flask import Flask, render_template, Response, jsonify
from keras.models import load_model

# ====================================================================
#                      SETUP AND GLOBAL VARIABLES
# ====================================================================
app = Flask(__name__)

# --- Recommendation Data (Unchanged) ---
recommendations = {
    "Oval": {"straight": "French crop", "wavy": "Textured crop", "curly": "Curly fringe", "kinky": "High top fade"},
    "Round": {"straight": "Quiff", "wavy": "Faux hawk", "curly": "Undercut with curly top", "kinky": "Clean lineup"},
    "Square": {"straight": "Crew cut", "wavy": "Side swept", "curly": "Short curly top with fade", "kinky": "Short afro"},
    "Heart": {"straight": "Side part", "wavy": "Loose wave", "curly": "Medium length curls", "kinky": "Tapered cut"},
    "Diamond": {"straight": "Fringe", "wavy": "Mid length waves", "curly": "Curly fringe", "kinky": "Buzzcut"},
    "Oblong": {"straight": "French crop", "wavy": "Messy waves (medium length)", "curly": "Curly fringe", "kinky": "Buzzcut"},
    "Triangle": {"straight": "Quiff", "wavy": "Textured crop", "curly": "Curly pompadour", "kinky": "High top fade"}
}

# --- MODEL AND LABELS ---
class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

# ---- THREADING VARIABLES ----
lock = threading.Lock()
output_frame = None
last_prediction = {"class": "Loading...", "confidence": 0.0}

# ====================================================================
#                       PREDICTION THREAD
# ====================================================================
def prediction_worker():
    # Load the model INSIDE this thread
    model = load_model("keras_Model.h5", compile=False)
    
    global last_prediction, output_frame
    while True:
        with lock:
            if output_frame is None: continue
            frame_to_process = output_frame.copy()

        image = cv2.resize(frame_to_process, (224, 224), interpolation=cv2.INTER_AREA)
        image_for_model = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_for_model = (image_for_model / 127.5) - 1
        prediction = model.predict(image_for_model, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # =======================================================================
        # THIS IS THE FINAL FIX: Convert numpy.float32 to a standard Python float
        # =======================================================================
        last_prediction = {"class": class_name[2:].strip(), "confidence": float(confidence_score)}
        
        time.sleep(0.1) 

# ====================================================================
#                       FLASK VIDEO STREAMING & ROUTES
# ====================================================================
def generate_frames():
    global output_frame
    while True:
        success, frame = camera.read()
        if not success: break
        current_class = last_prediction["class"]
        current_confidence = last_prediction["confidence"]
        BOX_COLOR = (0, 0, 0); TEXT_COLOR = (255, 255, 255); FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8; THICKNESS = 2; ALPHA = 0.5; PADDING = 15
        LABEL_COLON_GAP = 5; COLON_VALUE_GAP = 10
        label1_text = "Shape"; label2_text = "Confidence"; separator = ":"
        value1_text = current_class; value2_text = f"{int(np.round(current_confidence * 100))}%"
        (w_l1, h_l1), _ = cv2.getTextSize(label1_text, FONT, FONT_SCALE, THICKNESS)
        (w_l2, h_l2), _ = cv2.getTextSize(label2_text, FONT, FONT_SCALE, THICKNESS)
        max_label_width = max(w_l1, w_l2)
        (w_sep, h_sep), _ = cv2.getTextSize(separator, FONT, FONT_SCALE, THICKNESS)
        (w_v1, h_v1), _ = cv2.getTextSize(value1_text, FONT, FONT_SCALE, THICKNESS)
        (w_v2, h_v2), _ = cv2.getTextSize(value2_text, FONT, FONT_SCALE, THICKNESS)
        max_value_width = max(w_v1, w_v2)
        box_width = PADDING + max_label_width + LABEL_COLON_GAP + w_sep + COLON_VALUE_GAP + max_value_width + PADDING
        row1_height = max(h_l1, h_v1, h_sep); row2_height = max(h_l2, h_v2, h_sep)
        box_height = PADDING + row1_height + PADDING // 2 + row2_height + PADDING
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + box_width, 5 + box_height), BOX_COLOR, -1)
        frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
        x_col1_labels = 5 + PADDING; x_col2_separator = x_col1_labels + max_label_width + LABEL_COLON_GAP
        x_col3_values = x_col2_separator + w_sep + COLON_VALUE_GAP
        y_row1 = 5 + PADDING + row1_height
        y_row2 = y_row1 + PADDING // 2 + row2_height
        cv2.putText(frame, label1_text, (x_col1_labels, y_row1), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, separator, (x_col2_separator, y_row1), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, value1_text, (x_col3_values, y_row1), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, label2_text, (x_col1_labels, y_row2), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, separator, (x_col2_separator, y_row2), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, value2_text, (x_col3_values, y_row2), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        with lock: output_frame = frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', recommendations=recommendations)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with lock:
        return jsonify(last_prediction)

# ====================================================================
#                          MAIN EXECUTION
# ====================================================================
if __name__ == "__main__":
    prediction_thread = threading.Thread(target=prediction_worker, daemon=True)
    prediction_thread.start()
    app.run(debug=False, threaded=True)

camera.release()