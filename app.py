import os
import torch
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from transformers import pipeline
from pdf2image import convert_from_path
import io

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the trained model
def load_model():
    filename = 'model.pkl'  # Path to your model file
    global model
    try:
        with open(filename, 'rb') as f:
            model = torch.load(f, map_location=torch.device('cpu'))
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Function to handle different file types
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_file_path = os.path.join("temp", uploaded_file.name.replace('\\', '/'))

    os.makedirs("temp", exist_ok=True)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_extension == 'pdf':
        pages = convert_from_path(temp_file_path, 300)
        images = []
        for i, page in enumerate(pages):
            image_path = os.path.join("temp", f"page_{i}.jpg").replace('\\', '/')
            page.save(image_path, 'JPEG')
            images.append(image_path)
        return images
    elif file_extension in ['jpg', 'jpeg', 'png']:
        return [temp_file_path]
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image file.")

# Extract prescription details
def extract_prescription_details(image_path):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    extracted_text = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            extracted_text.append(text)
    
    return " ".join(extracted_text)

# Prescription review network
def review_prescription(prediction_confidence):
    threshold = 0.75  # Confidence threshold
    if prediction_confidence < threshold:
        return "Prescription requires pharmacist review. Sending to network."
    return "Prescription processed successfully."

# Save prescription data
def save_prescription_data(text, doc_type, confidence):
    data = {
        'Extracted Text': [text],
        'Document Type': [doc_type],
        'Confidence Score': [confidence]
    }
    df = pd.DataFrame(data)
    df.to_csv('prescription_data.csv', index=False)
    st.success("Extracted prescription saved to prescription_data.csv")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='prescription_data.csv',
        mime='text/csv',
    )

# Streamlit UI
def main():
    st.title("Handwritten Prescription Digitizer")
    
    doc_type = st.text_input("Enter document type", "Prescription")
    uploaded_file = st.file_uploader("Upload a prescription", type=['pdf', 'jpg', 'jpeg', 'png'])
    
    if st.button("Process Prescription"):
        if doc_type and uploaded_file:
            image_paths = handle_file_upload(uploaded_file)
            for image_path in image_paths:
                extracted_text = extract_prescription_details(image_path)
                confidence = 0.85  # Placeholder confidence score
                review_status = review_prescription(confidence)
                st.write(f"Extracted Text: {extracted_text}")
                st.write(f"Confidence Score: {confidence}")
                st.write(f"Review Status: {review_status}")
                save_prescription_data(extracted_text, doc_type, confidence)
        else:
            st.error("Please enter document type and upload a prescription.")
    
    # Pharmacist Dashboard
    st.sidebar.title("Pharmacist Dashboard")
    st.sidebar.write("View prescription review statistics")
    if os.path.exists('prescription_data.csv'):
        df = pd.read_csv('prescription_data.csv')
        st.sidebar.write(df)
    else:
        st.sidebar.write("No data available.")
    
    load_model()

if __name__ == "__main__":
    main()
