import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image

st.set_page_config(page_title="Abhedya Security System", layout="wide")

try:
    from ultralytics import YOLO
    from unattended import UnattendedObjectDetector
    from restricted_intrusion import detect_intrusion
    from overcrowding import detect_overcrowding
    modules_imported = True
except ImportError as e:
    modules_imported = False
    st.warning(f"Some detection modules could not be imported: {str(e)}")

if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'polygon_complete' not in st.session_state:
    st.session_state.polygon_complete = False
if 'detector_instances' not in st.session_state:
    st.session_state.detector_instances = {}
if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False

@st.cache_resource
def get_model():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8s.pt")
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

def draw_polygon(frame, roi_points, scale_ratio=1.0):
    if not roi_points or len(roi_points) < 2:
        return frame
        
    scaled_points = [(int(p[0] * scale_ratio), int(p[1] * scale_ratio)) for p in roi_points]
    
    try:
        overlay = frame.copy()
        cv2.polylines(overlay, [np.array(scaled_points)], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.fillPoly(overlay, [np.array(scaled_points)], color=(0, 165, 255))
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
        for pt in scaled_points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    except Exception as e:
        st.error(f"Error drawing polygon: {str(e)}")
    
    return frame

def resize_frame_with_aspect_ratio(frame, scale_ratio=0.7):   
    height, width = frame.shape[:2]
    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

def process_video_file(video_path, use_unattended, use_intrusion, use_overcrowding, 
                      unattended_time, crowd_threshold, roi_points):
    if not os.path.exists(video_path):
        st.error("Video file not found")
        return
        
    # model
    model = get_model() if modules_imported else None
    if model is None and (use_intrusion or use_overcrowding or use_unattended):
        st.error("YOLO model could not be loaded")
        return
    
    # UnattendedObjectDetector
    unattended_detector = None
    if use_unattended and modules_imported:
        if 'unattended' not in st.session_state.detector_instances:
            try:
                st.session_state.detector_instances['unattended'] = UnattendedObjectDetector(model_path="yolov8s.pt")
            except Exception as e:
                st.error(f"Error initializing unattended object detector: {str(e)}")
                return
        unattended_detector = st.session_state.detector_instances['unattended']
        unattended_detector.UNATTENDED_TIMEOUT = unattended_time
    
    #in video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return
    
    #vidproperties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # scale dimensions
    scale_ratio = 1.0
    scaled_width = int(width * scale_ratio)
    scaled_height = int(height * scale_ratio)
    
    #placeholder for the video
    video_placeholder = st.empty()
    
    stop_col = st.container()
    stop_button = stop_col.button("Stop Processing", key="stop_button")

    st.session_state.stop_processing = False
    
    frame_count = 0
    while cap.isOpened() and not st.session_state.stop_processing:
        ret, frame = cap.read()
        if not ret:
            st.info("End of video reached")
            break
            
        frame = resize_frame_with_aspect_ratio(frame, scale_ratio)

        frame_count += 1
        if frame_count % max(1, int(fps/10)) != 0:
            continue
        
        try:
            if use_unattended and unattended_detector:
                frame = unattended_detector.detect_unattended_objects(frame)
                
            if use_intrusion and len(roi_points) > 2 and 'detect_intrusion' in globals():
                # Scale ROI points to match resized frame
                scaled_roi_points = [(int(p[0] * scale_ratio), int(p[1] * scale_ratio)) for p in roi_points]
                frame = detect_intrusion(frame, model, scaled_roi_points)
                
            if use_overcrowding and 'detect_overcrowding' in globals():
                frame = detect_overcrowding(frame, model, crowd_threshold)
        except Exception as e:
            st.error(f"Error in detection: {str(e)}")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        time.sleep(0.1)
        
        if stop_button:
            st.session_state.stop_processing = True
            break
    cap.release()

def main():
    st.title("ABHEDYA : AI Surveillance System")
    st.sidebar.title("Configuration")

    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], key="video_uploader")
    
    #options
    st.sidebar.header("Detection Options")
    use_unattended = st.sidebar.checkbox("Unattended Object Detection", value=True)
    use_intrusion = st.sidebar.checkbox("Restricted Area Intrusion")
    use_overcrowding = st.sidebar.checkbox("Overcrowding Detection")
    
    #Thresholds
    st.sidebar.header("Detection Parameters")
    unattended_time = st.sidebar.slider("Unattended Object Time (seconds)", 5, 200, 10) if use_unattended else 10
    crowd_threshold = st.sidebar.slider("Overcrowding Threshold (people)", 2, 50, 10) if use_overcrowding else 10
    
    #Video display area
    video_container = st.container()
    video_placeholder = video_container.empty()
    
    #Reset ROIbutton
    if use_intrusion:
        if st.sidebar.button("Reset ROI Points"):
            st.session_state.roi_points = []
            st.session_state.polygon_complete = False
    
    #file upload
    if uploaded_file is not None:
        try:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            
            st.success("Video uploaded successfully!")
            
            # Open video for metadata
            cap = cv2.VideoCapture(st.session_state.video_path)
            
            if not cap.isOpened():
                st.error("Error opening video file")
                return
 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            scale_ratio = 1.0
            scaled_width = int(width * scale_ratio)
            scaled_height = int(height * scale_ratio)
            
            
            st.sidebar.write(f"Original Resolution: {width}x{height}")
            st.sidebar.write(f"Scaled Resolution: {scaled_width}x{scaled_height}")
            st.sidebar.write(f"FPS: {fps}")
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = resize_frame_with_aspect_ratio(frame, scale_ratio)
                
                preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if st.session_state.roi_points:
                    preview_frame = draw_polygon(preview_frame.copy(), st.session_state.roi_points, scale_ratio)
               
                video_placeholder.image(preview_frame, caption="Video Preview", use_column_width=True)
            
            # ROI definition
            if use_intrusion and not st.session_state.polygon_complete:
                st.subheader("Define Restricted Area")
                col1, col2 = st.columns(2)
                x = col1.number_input("X Coordinate", 0, width, step=1, key="roi_x")
                y = col2.number_input("Y Coordinate", 0, height, step=1, key="roi_y")
                
                col1, col2 = st.columns(2)
                if col1.button("Add Point", key="add_point_button"):
                    st.session_state.roi_points.append((int(x), int(y)))
                    
                    # Update preview with ROI points
                    if ret:
                        resized_frame = resize_frame_with_aspect_ratio(frame.copy(), scale_ratio)
                        preview_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        preview_frame = draw_polygon(preview_frame, st.session_state.roi_points, scale_ratio)
                        video_placeholder.image(preview_frame, caption="Video Preview with ROI", use_column_width=True)
                
                if len(st.session_state.roi_points) > 2 and col2.button("Complete Polygon", key="complete_polygon_button"):
                    st.session_state.polygon_complete = True
                    
                    # Update preview with completed ROI
                    if ret:                       
                        resized_frame = resize_frame_with_aspect_ratio(frame.copy(), scale_ratio)
                        preview_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        preview_frame = draw_polygon(preview_frame, st.session_state.roi_points, scale_ratio)
                        video_placeholder.image(preview_frame, caption="Video Preview with ROI", use_column_width=True)
                
                # Display current points
                if st.session_state.roi_points:
                    st.write("Current points:", st.session_state.roi_points)
            
            # Process video button
            if st.button("Process Video", key="process_video_button"):
                process_video_file(
                    st.session_state.video_path,
                    use_unattended,
                    use_intrusion,
                    use_overcrowding,
                    unattended_time,
                    crowd_threshold,
                    st.session_state.roi_points
                )
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    else:
        st.info("Please upload a video file")
    
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        if st.sidebar.button("Clear uploaded video"):
            try:
                os.unlink(st.session_state.video_path)
                st.session_state.video_path = None
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing video: {str(e)}")

if __name__ == "__main__":
    main()
