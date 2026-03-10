import streamlit as st
from PIL import Image
import os
import cv2
import torch
import pandas as pd
import easyocr
from io import BytesIO
from Classification import predict_and_correct_images, model, data_transforms  # Import model and helper functions
import subprocess

# python -m streamlit run app.py

# Function to get a unique file name for the CSV
def get_unique_csv_name(base_folder, base_name="ocr_results", extension=".csv"):
    """
    Generate a unique CSV file name like 'ocr_results1.csv', 'ocr_results2.csv', etc.
    """
    counter = 1
    while True:
        file_name = f"{base_name}{counter}{extension}"
        file_path = os.path.join(base_folder, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def get_latest_exp_folder(base_folder="./runs/detect"):
    """
    Get the latest 'exp' folder in the specified base folder.
    :param base_folder: The parent folder containing 'exp' folders.
    :return: Path to the latest 'exp' folder, or None if no folder exists.
    """
    if not os.path.exists(base_folder):
        return None

    # List all subdirectories in the base folder
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir() and f.name.startswith("exp")]
    
    if not subfolders:
        return None

    # Sort subfolders by their creation time (newest first)
    latest_folder = max(subfolders, key=os.path.getmtime)
    return latest_folder


# 设置页面标题和侧边栏
st.title("Receipt Processing System")
st.sidebar.title("Function selection")
functionality = st.sidebar.radio(
    "Please select the function",
    ("Classification", "Detection", "OCR")
)

# 全局变量
uploaded_images = []  # 存储上传的图片

# 分类检测
if functionality == "Classification":
    st.header("Step 1: Classification and rotation if needed")
    
    # Upload multiple images
    uploaded_files = st.file_uploader("Upload the image", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Display uploaded images
        st.subheader("Upload image")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)

        if st.button("Start the Classification"):
            # Call `predict_and_correct_images` with in-memory uploaded files
            st.subheader("Classification results")
            for uploaded_file in uploaded_files:
                # Open the image using PIL
                image = Image.open(uploaded_file)
                
                # Call the classification function
                corrected_image, predicted_label = predict_and_correct_images(
                    [image],  # Pass image directly
                    model=model,
                    transform=data_transforms
                )
                # Display the corrected image
                st.image(corrected_image[0], caption=f"{uploaded_file.name} (Predicted: {predicted_label[0]})", width=400)

# Object Detection
elif functionality == "Detection":
    st.header("Step 2: Object Detection on Corrected Images")

    detection_folder = "./data/corrected_images"  # Input folder for object detection
    base_output_folder = "./runs/detect"  # Base folder for detection results

    # Check if input folder exists and contains images
    if os.path.exists(detection_folder) and os.listdir(detection_folder):
        if st.button("Start Object Detection"):
            # Construct the command to run detect.py
            command = [
                "python", "detect.py",
                "--img", "640",
                "--conf", "0.5",
                "--device", "cpu",
                "--weights", "runs/train/exp/weights/best.pt",
                "--source", detection_folder,
                "--save-txt", "--save-conf"
            ]

            # Run the detect.py script
            try:
                with st.spinner("Running object detection... Please wait."):
                    subprocess.run(command, check=True)
                st.success("Object detection completed!")

                # Get the latest exp folder
                latest_output_folder = get_latest_exp_folder(base_output_folder)

                if latest_output_folder:
                    st.subheader("Detection Results")

                    # Display all detected images
                    detected_images = [f for f in os.listdir(latest_output_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                    if detected_images:
                        for file_name in detected_images:
                            detected_image_path = os.path.join(latest_output_folder, file_name)
                            st.image(detected_image_path, caption=f"Detected Image: {file_name}", width=400)
                            # Optional: Print the path of the detected image
                            st.text(f"Path: {detected_image_path}")
                    else:
                        st.warning("No detected images found in the latest folder.")
                else:
                    st.warning("No detection output folder found. Please check detect.py.")
            except subprocess.CalledProcessError as e:
                st.error(f"Object detection failed: {e}")
    else:
        st.warning("No corrected images found. Please perform image classification first.")
        

elif functionality == "OCR":
    st.header("Step 3: Perform OCR on Detected Areas")
    
    # Dynamically locate the latest detection folder
    detection_folder = get_latest_exp_folder(base_folder="./runs/detect")

    if st.button("Start OCR"):
        # Construct the command to run OCR.py
        command = [
            "python", "OCR.py",
            "--input_folder", detection_folder,
            "--image_dir", "./data/corrected_images",
            "--output_csv", "ocr_results"  # Base name for the output CSV
        ]

        try:
            # Run OCR.py
            with st.spinner("Performing OCR... Please wait."):
                subprocess.run(command, check=True)

            # Look for the latest CSV in the root directory
            import glob
            csv_files = glob.glob("./ocr_results*.csv")
            if csv_files:
                ocr_csv_path = max(csv_files, key=os.path.getmtime)  # Get the latest file
                st.success("OCR completed! The results are saved as a CSV.")
                st.download_button(
                    label="Download OCR Results",
                    data=open(ocr_csv_path, "rb"),
                    file_name=os.path.basename(ocr_csv_path),
                    mime="text/csv",
                )
            else:
                st.warning("The OCR results CSV file was not found.")
        except subprocess.CalledProcessError as e:
            st.error(f"OCR failed: {e}")
        

        
# elif functionality == "OCR":
#     st.header("Step 3: Perform OCR on Detected Areas")

#     # Dynamically locate the latest detection folder
#     detection_folder = get_latest_exp_folder(base_folder="./runs/detect")

#     if detection_folder:
#         if st.button("Start OCR"):
#             # Construct the command to run OCR.py
#             command = [
#                 "python", "OCR.py", 
#                 "--input_folder", detection_folder,
#                 "--image_dir", "./data/corrected_images",  # Optional: Adjust if needed
#                 "--output_csv", "ocr_results_easyocr.csv"  # Output file name
#             ]

#             # Run OCR.py
#             try:
#                 with st.spinner("Performing OCR... Please wait."):
#                     subprocess.run(command, check=True)
#                 st.success("OCR completed! The results are saved as a CSV.")

#                 # Add a download button for the output CSV
#                 ocr_csv_path = os.path.join(".", "ocr_results*.csv")
#                 if os.path.exists(ocr_csv_path):
#                     st.download_button(
#                         label="Download OCR Results",
#                         data=open(ocr_csv_path, "rb"),
#                         file_name="ocr_results_easyocr.csv",
#                         mime="text/csv",
#                     )
#                 else:
#                     st.warning("The OCR results CSV file was not found.")
#             except subprocess.CalledProcessError as e:
#                 st.error(f"OCR failed: {e}")
#     else:
#         st.warning("No detection folder found. Please perform object detection first.")

