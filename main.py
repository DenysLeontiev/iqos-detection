import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import time

st.title("Iqos lid segmentation")

# Simplified mode selection
mode = st.radio("Select Mode", ["Upload File", "Camera"], horizontal=True)

confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Load model
@st.cache_resource
def load_model():
    model = YOLO("100-epochs-640-352-imgsz-yes-augmentation-yolo11n-seg.pt")
    return model

model = load_model()

# Camera mode - Continuous photo capture with auto-refresh
if mode == "Camera":
    st.write("### 📷 Camera - Real-Time Segmentation")
    st.info("📱 Take photos continuously to see segmentation results")
    
    # Use camera input without forced refresh
    camera_photo = st.camera_input("Take a photo to segment")
    
    if camera_photo is not None:
        # Convert image
        image = Image.open(camera_photo)
        img_array = np.array(image)
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Captured Image**")
            st.image(image, use_container_width=True)
        
        # Run inference
        start_time = time.time()
        results = model(img_array, conf=confidence_threshold, imgsz=320, device='cpu', verbose=False)
        inference_time = time.time() - start_time
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.write("**Segmentation Result**")
            st.image(annotated_frame_rgb, use_container_width=True)
        
        # Show FPS
        fps = 1 / inference_time if inference_time > 0 else 0
        st.success(f"✅ Processed in {inference_time:.2f}s ({fps:.1f} FPS)")
        
        st.info("💡 Tip: Click the camera button again to capture and segment another frame")

# File upload mode
elif mode == "Upload File":
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ["jpg", "jpeg", "png"]:
                # Load and optimize image for phone uploads
                image = Image.open(uploaded_file)
                
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                # Resize if image is too large (phone images can be huge)
                max_size = 1920
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    st.info(f"Image resized to {image.size[0]}x{image.size[1]} for faster processing")
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Run inference
                with st.spinner("Processing..."):
                    results = model(image, conf=confidence_threshold)
                
                st.image(results[0].plot(), caption="Detection Result", use_column_width=True)
            
            elif file_extension in ["mp4", "avi", "mov"]:
                frame_interval = st.slider("Process every N-th frame", min_value=1, max_value=30, value=1)
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
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please try a different image or check the file format.")