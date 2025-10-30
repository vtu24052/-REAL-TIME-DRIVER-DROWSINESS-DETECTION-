import streamlit as st
import time
import numpy as np
import cv2
from io import BytesIO

# Assuming you have constants or parameters defined elsewhere
EYE_OPEN_RATE_FPS = 0.3
EYE_CLOSED_RATE_FPS = 0.6
TRUE = True
FALSE = False
input_shape = (480, 640, 3)  # Set to your frame dimensions
result_shape = (6,)  # Change this as per the bounding box structure

# Initialize session state variables
if "smemory_is_drowsy" not in st.session_state:
    st.session_state.smemory_is_drowsy = FALSE
if "smemory_eyeopen" not in st.session_state:
    st.session_state.smemory_eyeopen = 0
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

def process_frame(frame, eye_open_count, eye_closed_count):
    # Simulate detection logic and update `eye_open_count` and `eye_closed_count` for simplicity
    # Example dummy logic: detect based on color or threshold (replace with actual model)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    eye_open_count += np.sum(threshold > 128) / 1000  # Example metric
    eye_closed_count += np.sum(threshold <= 128) / 1000  # Example metric

    return eye_open_count, eye_closed_count

def drowsiness_check(eye_open_count, eye_closed_count):
    if eye_closed_count / (eye_open_count + eye_closed_count) > EYE_CLOSED_RATE_FPS:
        st.session_state.smemory_is_drowsy = TRUE
    else:
        st.session_state.smemory_is_drowsy = FALSE

def main():
    st.title("Drowsiness Detection System")
    
    # Upload video file or use camera
    video_source = st.sidebar.selectbox("Video Source", ["Upload", "Camera"])
    if video_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi"])
        if uploaded_file is not None:
            # Convert file to opencv-compatible format
            video_file = BytesIO(uploaded_file.read())
            cap = cv2.VideoCapture(video_file)
    else:
        cap = cv2.VideoCapture(0)

    frame_display = st.empty()
    drowsiness_text = st.empty()
    
    eye_open_count = 0
    eye_closed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for drowsiness
        eye_open_count, eye_closed_count = process_frame(frame, eye_open_count, eye_closed_count)
        drowsiness_check(eye_open_count, eye_closed_count)

        # Overlay drowsiness warning on frame if drowsy
        if st.session_state.smemory_is_drowsy:
            cv2.putText(frame, 'Drowsiness Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            drowsiness_text.warning("Drowsiness detected! Please stay alert.")
        else:
            drowsiness_text.info("Driver is alert.")

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, channels="RGB")

        # Limit frame rate to mimic real-time processing
        time.sleep(0.1)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
