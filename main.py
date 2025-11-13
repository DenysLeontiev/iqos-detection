import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import time

st.title("YOLOv11 Segmentation Demo")

# Add mode selection
mode = st.radio("Select Mode", ["Upload File", "Phone Camera (Live)", "Webcam (Desktop)"], horizontal=True)

if mode == "Upload File":
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
else:
    uploaded_file = None

frame_interval = st.slider("Process every N-th frame (for videos)", min_value=1, max_value=30, value=1)
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Phone Camera mode - uses browser camera directly
if mode == "Phone Camera (Live)":
    st.write("### 📱 Phone Camera - Real-Time Segmentation")
    st.info("✅ This works directly on your phone! Just allow camera access when prompted.")
    
    # Performance settings
    col1, col2 = st.columns(2)
    with col1:
        img_size = st.selectbox("Image Size (smaller = faster)", [256, 320, 416, 640], index=1)
    with col2:
        process_interval = st.selectbox("Process every N frames", [1, 2, 3], index=0)
    
    # Load model
    @st.cache_resource
    def load_model():
        model = YOLO("100-epochs-640-352-imgsz-yes-augmentation-yolo11n-seg.pt")
        return model
    
    model = load_model()
    
    # Camera input with automatic refresh
    camera_photo = st.camera_input("Take a photo or enable continuous mode")
    
    # Add continuous mode
    continuous_mode = st.checkbox("🔄 Continuous Mode (for real-time video)", value=False)
    
    if camera_photo is not None:
        # Convert image
        image = Image.open(camera_photo)
        img_array = np.array(image)
        
        # Display original
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original**")
            st.image(image, use_column_width=True)
        
        # Run inference
        start_time = time.time()
        results = model(img_array, conf=confidence_threshold, imgsz=img_size, device='cpu', verbose=False)
        inference_time = time.time() - start_time
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display result
        with col2:
            st.write("**Segmentation Result**")
            st.image(annotated_frame_rgb, use_column_width=True)
        
        # Show performance
        fps = 1 / inference_time if inference_time > 0 else 0
        st.metric("Processing Speed", f"{fps:.1f} FPS")
        
        if continuous_mode:
            st.info("📸 Take another photo to update (automatic refresh coming soon)")
            # Note: True continuous mode requires JavaScript/custom component
            st.rerun()

# Desktop Webcam mode - for computers with webcam
elif mode == "Webcam (Desktop)":
    st.write("### 💻 Desktop Webcam - Real-Time Segmentation")
    st.warning("⚠️ This mode is for desktop/laptop webcams only. For phones, use 'Phone Camera (Live)' mode.")
    
    # Camera settings
    col1, col2 = st.columns(2)
    with col1:
        camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0)
    with col2:
        target_fps = st.selectbox("Target FPS", [10, 15, 20, 30], index=1)
    
    # Performance settings
    st.write("#### Performance Settings")
    col3, col4 = st.columns(2)
    with col3:
        img_size = st.selectbox("Image Size (smaller = faster)", [256, 320, 416, 640], index=1)
    with col4:
        use_gpu = st.checkbox("Use GPU (if available)", value=False)
    
    start_camera = st.button("Start Camera", type="primary")
    stop_camera = st.button("Stop Camera", type="secondary")
    
    if start_camera:
        st.session_state.camera_running = True
    if stop_camera:
        st.session_state.camera_running = False
    
    if st.session_state.get('camera_running', False):
        # Load model with optimizations
        @st.cache_resource
        def load_model():
            model = YOLO("100-epochs-640-352-imgsz-yes-augmentation-yolo11n-seg.pt")
            return model
        
        model = load_model()
        
        # Check GPU availability
        import torch
        gpu_available = torch.cuda.is_available() if use_gpu else False
        device = 'cuda:0' if gpu_available else 'cpu'
        
        if use_gpu and not gpu_available:
            st.warning("GPU requested but not available. Using CPU instead.")
        
        st.info(f"Running on: {device.upper()}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
        
        if not cap.isOpened():
            st.error(f"Cannot open camera {camera_id}. Please check camera ID.")
            st.session_state.camera_running = False
        else:
            st.success(f"Camera opened successfully!")
            
            # Create placeholder for video feed
            frame_placeholder = st.empty()
            fps_placeholder = st.empty()
            
            # FPS calculation
            fps_counter = []
            frame_count = 0
            
            try:
                while st.session_state.get('camera_running', False):
                    start_time = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame")
                        break
                    
                    # Run inference with optimizations
                    results = model(
                        frame,
                        conf=confidence_threshold,
                        imgsz=img_size,
                        device=device,
                        verbose=False  # Disable verbose output for speed
                    )
                    
                    # Get annotated frame
                    annotated_frame = results[0].plot()
                    
                    # Convert BGR to RGB for display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Calculate FPS
                    end_time = time.time()
                    fps = 1 / (end_time - start_time)
                    fps_counter.append(fps)
                    if len(fps_counter) > 30:
                        fps_counter.pop(0)
                    avg_fps = sum(fps_counter) / len(fps_counter)
                    
                    # Add FPS text to frame
                    cv2.putText(
                        annotated_frame_rgb,
                        f"FPS: {avg_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display frame
                    frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Display FPS info
                    fps_placeholder.metric("Average FPS", f"{avg_fps:.1f}")
                    
                    frame_count += 1
                    
                    # Small delay to prevent UI blocking
                    time.sleep(0.001)
                    
            except Exception as e:
                st.error(f"Error during camera processing: {str(e)}")
            finally:
                cap.release()
                st.session_state.camera_running = False
                st.info("Camera stopped")

# File upload mode
elif uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    model = YOLO("100-epochs-640-352-imgsz-yes-augmentation-yolo11n-seg.pt")
    
    if file_extension in ["jpg", "jpeg", "png"]:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        image = Image.open(uploaded_file)
        
        results = model(image, conf=confidence_threshold)
        
        st.image(results[0].plot(), caption="Detection Result", use_column_width=True)
    
    elif file_extension in ["mp4", "avi", "mov"]:
        st.video(uploaded_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name
        
        # Process video
        st.write(f"Processing video (every {frame_interval} frame(s))...")
        
        cap = cv2.VideoCapture(tmp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video with H264 codec
        output_path = tmp_video_path.replace(f'.{file_extension}', '_output.mp4')
        
        # Try different codecs for compatibility
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID
        ]
        
        out = None
        for fourcc in fourcc_options:
            out = cv2.VideoWriter(output_path, fourcc, fps / frame_interval, (width, height))
            if out.isOpened():
                break
        
        if not out or not out.isOpened():
            st.error("Could not create video writer. Please try a different video format.")
            cap.release()
            os.unlink(tmp_video_path)
        else:
            frame_count = 0
            progress_bar = st.progress(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every n-th frame
                if frame_count % frame_interval == 0:
                    # Run inference
                    results = model(frame, conf=confidence_threshold)
                    
                    # Get annotated frame
                    annotated_frame = results[0].plot()
                    
                    # Ensure frame is in correct format
                    if annotated_frame.dtype != np.uint8:
                        annotated_frame = annotated_frame.astype(np.uint8)
                    
                    # Write to output video
                    out.write(annotated_frame)
                    
                    # Update progress
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                frame_count += 1
            
            cap.release()
            out.release()
            
            progress_bar.progress(1.0)
            
            # Display output video
            st.success("Processing complete!")
            
            # Read and display the video
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                    
                # Offer download
                st.download_button(
                    label="Download processed video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("Failed to create output video.")
            
            # Cleanup temp files
            try:
                os.unlink(tmp_video_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass