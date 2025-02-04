!pip install pytesseract opencv-python
!apt update && apt install -y tesseract-ocr

from PIL import Image
import pytesseract
import pandas as pd
import cv2
import numpy as np

# Preprocess the image for better OCR accuracy
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edges = cv2.Canny(gray, 30, 200)  # Edge detection
    return Image.fromarray(gray)

# Load and preprocess the image
image_path = "/content/sms-csv-file.png"
image = preprocess_image(image_path)

# Perform OCR
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(image, config=custom_config)

# Display extracted text
print(text)

# Process text into structured data
rows = [list(filter(None, row.split())) for row in text.splitlines() if row.strip()]

# Convert to DataFrame and export to CSV
df = pd.DataFrame(rows)
df.to_csv("output.csv", index=False)

print("CSV file has been saved as output.csv")
