#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 14:04:07 2025

@author: ayechanaung
"""
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import pickle

with open('model.pkl','rb') as f:
    model = pickle.load(f)
    
st.title("Image Classification with MobileNetV2 by Nongnuch Ketui")

upload_file = st.file_uploader("Upload image:",type=["jpg","jpeg","png"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption="")
    
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    
    preds = model.predict
    top_preds = decode_predictions(preds, top=3)[0]
    
    st.subheader("Prediction:")
    for i, pred in enumerate(top_preds) :
        st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%")
