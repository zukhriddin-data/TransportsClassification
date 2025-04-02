import streamlit as st
import plotly.express as px
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Transportni klassifikatsiya qiluvchi model')

# rasm joylash
file = st.file_uploader('Rasm yuklash', type = ['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    img = PILImage.create(file)
# model
    model = load_learner('transport_model.pkl')

# prediction
    pred, pred_id, probs = model.predict(img) 
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')


    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
