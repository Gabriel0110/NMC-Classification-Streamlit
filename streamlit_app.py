import streamlit as st
import streamlit.components.v1 as components
from image_classification import classification
from tensorflow.keras.preprocessing import image


st.title("NUT Midline Carcinoma Pathology Classification")
st.text("Upload an image of cancer pathology to attempt to predict if it is NMC or not.")
st.write("")
st.write("")

uploaded_file = st.file_uploader("Choose a pathology image...")
if uploaded_file is not None:
    #image = Image.open(uploaded_file)
    img = image.load_img(uploaded_file, target_size=(224, 224, 3), color_mode="rgb")
    st.image(img, caption='Uploaded pathology image', use_column_width=True)
    st.write("")

    if st.button("Classify"):
        st.write("Classifying...")
        result = classification(img)
        if result < 0.5:
            st.write(f"Prediction score: {(1 - result) * 100:.2f}% - this pathology is predicted to be NMC.")
        else:
            st.write(f"Prediction score: {(1 - result) * 100:.2f}% - this pathology is NOT predicted to be NMC.")

st.write("")
st.write("")
st.write("")
st.write("")
st.markdown("<div style='text-align: center'><i> Model used: ResNet50 </i></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center'><i> Model accuracy at training-time: 0.8271 </i></div>", unsafe_allow_html=True)