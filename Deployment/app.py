import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np


from tensorflow.keras.models import load_model

st.set_page_config(page_title = 'News Category Prediction',
                  layout = "wide",
                  initial_sidebar_state = "expanded",
                  menu_items = {
                      'About' : 'News Category Prediction '
                  })


# load model
model = keras.models.load_model("model_milestone2")

#def predict(inputs):
    #y_pred = model.predict([inputs])
    #y_pred = y_pred.argmax(axis=1)
    #print(y_pred)
    #return y_pred.item()

label = ['Berita Biasa/Politik', 'Berita Hot/Selebritis', 'Berita Keuangan/Finance', 'Berita Travel', 'Berita Internet/Teknologi' 'Berita Kesehatan', 'Berita Otomotif', 'Berita Makanan/Minuman', 'Berita Olahraga']

st.title("Indonesian News Category Prediction")


news_title = st.text_input('Enter a News Title')
new_data = pd.DataFrame([news_title])
res = model.predict(new_data)
res = res.argmax()
press = st.button('Predict')
if press:
   st.title(label[res])



