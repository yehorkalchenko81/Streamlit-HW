import streamlit as st
import numpy as np
import io
from PIL import Image
from yolov5.detect import run
import os
import matplotlib.pyplot as plt

model_type = st.radio('Select model:', ['YOLOv5s', 'YOLOv5x'])
uploaded_file = st.file_uploader('Select photo for object detection', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.write("Назва файлу:", uploaded_file.name)
    file_contents = uploaded_file.read()
    image = Image.open(io.BytesIO(file_contents))
    image.save('photo.png')

    run(weights=f'{model_type.lower()}.pt', source='photo.png', save_txt=True)

    list_of_res = os.listdir('runs/detect')
    list_of_res.sort(key=lambda x: int(x[3:]) if x[3:] else 0)
    last_result_dir = f'runs/detect/{list_of_res[-1]}'
    
    last_res = Image.open(f'{last_result_dir}/photo.png')
    st.image(last_res, caption=f'Predicted by {model_type}', use_column_width=True)

    confidences = []

    for filename in os.listdir(f'{last_result_dir}/labels'):
        if filename.endswith('.txt'):  # зазвичай, файл з метками
            with open(os.path.join(f'{last_result_dir}/labels', filename)) as f:
                for line in f:
                    confidence = float(line.split()[1])
                    confidences.append(confidence)
    if confidences:
        fig, ax = plt.subplots()
        ax.hist(confidences, bins=20)
        ax.set_title('Distribution of Confidence Scores')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        
        st.pyplot(fig)
    else:
        st.write("No confidence scores found in results.")
