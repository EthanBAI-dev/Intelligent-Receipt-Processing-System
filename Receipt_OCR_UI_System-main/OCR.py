import os
import pandas as pd
import easyocr
import cv2

# EasyOCR reader initialization
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)


def get_sequential_csv_name(base_name="ocr_results", directory=".", extension=".csv"):
    """
    Generate a unique sequential file name like 'ocr_results1.csv', 'ocr_results2.csv', etc.
    :param base_name: The base name of the file (without extension).
    :param directory: The directory where the file will be saved.
    :param extension: The file extension (default: '.csv').
    :return: Full path to a unique sequential file name.
    """
    counter = 1
    while True:
        file_name = f"{base_name}{counter}{extension}"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1


def perform_ocr_on_folder(label_dir, image_dir, output_csv_base):
    """
    Perform OCR on detected regions based on labels in the specified folder.
    :param label_dir: Path to the folder containing detection result TXT files.
    :param image_dir: Path to the folder containing original images.
    :param output_csv_base: Base name for the output CSV file (without extension).
    """
    results = []

    # Ensure the label directory exists
    if not os.path.exists(label_dir):
        print(f"Label directory does not exist: {label_dir}")
        return

    # Ensure the image directory exists
    if not os.path.exists(image_dir):
        print(f"Image directory does not exist: {image_dir}")
        return

    # Process each label file
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            image_name = label_file.replace(".txt", ".jpg")  # Assume images are in JPG format
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Read label file
            with open(label_path, "r") as f:
                lines = f.readlines()

            # Initialize class results
            ocr_results = {"image_name": image_name, "class_0": "", "class_1": "", "class_2": ""}

            # Perform OCR on detected regions
            for line in lines:
                components = line.strip().split()
                class_id = int(components[0])
                x_center, y_center, width, height = map(float, components[1:5])

                # Convert normalized coordinates to absolute pixel coordinates
                h_original, w_original = image.shape[:2]
                x1 = int((x_center - width / 2) * w_original)
                y1 = int((y_center - height / 2) * h_original)
                x2 = int((x_center + width / 2) * w_original)
                y2 = int((y_center + height / 2) * h_original)

                # Modify bounding box for class_1 to extract the right 40%
                if class_id == 1:
                    x1 = int(x1 + (x2 - x1) * 0.6)  # Keep only the right 40%

                # Crop the detected area
                cropped_image = image[y1:y2, x1:x2]

                # Perform OCR on the cropped image
                try:
                    ocr_result = reader.readtext(cropped_image)
                    extracted_text = " ".join([res[1] for res in ocr_result])
                except Exception as e:
                    print(f"OCR failed for {label_file}: {e}")
                    extracted_text = ""

                # Assign the extracted text to the appropriate class
                if class_id == 0:
                    ocr_results["class_0"] += extracted_text + " "
                elif class_id == 1:
                    ocr_results["class_1"] += extracted_text + " "
                elif class_id == 2:
                    ocr_results["class_2"] += extracted_text + " "

            # Clean up class texts and add to results
            ocr_results["class_0"] = ocr_results["class_0"].strip()
            ocr_results["class_1"] = ocr_results["class_1"].strip()
            ocr_results["class_2"] = ocr_results["class_2"].strip()
            results.append(ocr_results)

    # Save OCR results to a CSV file in the root directory
    if results:
        df = pd.DataFrame(results)
        # Ensure column order: image_name, class_1, class_0, class_2
        df = df[["image_name", "class_1", "class_0", "class_2"]]

        # Generate a unique sequential file name in the root directory
        output_csv = get_sequential_csv_name(base_name=output_csv_base, directory=".")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"OCR results saved to {output_csv}")