#========================import packages=========================================================
import streamlit as st
import numpy as np
import re
import os
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


Main_dir = os.getcwd()
Image_data_dir = os.path.join(Main_dir, "Image_data")
allowed_extensions = ["jpeg" ,"png" ,"jpg", "bmp"]
IMAGE_SIZE = 155

image_model = load_model(os.path.join(Main_dir,'models','image_model.h5'))
IMAGE_CLASS_NAMES = pickle.load(open(os.path.join(Main_dir,'models','image_classes.pkl'),'rb'))


def predict_image_emotion(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0  # Normalize properly to [0, 1]
    
    
    image_batch = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_batch)
    result = np.argmax(predictions)
    
    print(f"Predicted emotion is {IMAGE_CLASS_NAMES[result]}")
    return IMAGE_CLASS_NAMES[result]

st.title("Human Emotion Detector App")
st.write("This app predicts the emotion of a person from a text or an image")

st.write("## Image Emotion Prediction")

image_file = st.file_uploader("Upload an image", type=allowed_extensions)
if image_file is not None:
    # Save the uploaded image to a temporary file
    temp_image_path = os.path.join(Image_data_dir, "temp_image." + image_file.name.split('.')[-1])
    with open(temp_image_path, "wb") as f:
        f.write(image_file.getbuffer())
    
    st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Emotion from Image"):
        predicted_emotion = predict_image_emotion(temp_image_path, image_model)
        st.success(f"Predicted Emotion: {predicted_emotion}")
else:
    st.write("Please upload an image to predict emotion.")
    
    
    






import nltk
from nltk.corpus import stopwords
nltk.download('stopwords',quiet=True)
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()



text_Encoder = pickle.load(open(os.path.join(Main_dir,'models','text_Encoder.pkl'),'rb'))
tokenizer = pickle.load(open(os.path.join(Main_dir,'models','tokenizer.pkl'),'rb'))
MAX_LEN = pickle.load(open(os.path.join(Main_dir,'models','MAX_LEN.pkl'),'rb'))


lr_best_model = pickle.load(open(os.path.join(Main_dir,'models','lr_best_model.pkl'),'rb'))
grumodel = load_model(os.path.join(Main_dir, 'models', 'grumodel.keras'), compile=False)
lstmmodel = load_model(os.path.join(Main_dir, 'models', 'lstmmodel.keras'), compile=False)

    
def clean_content(text):
    
    text = re.sub(r'[^a-zA-Z]',' ',text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.lower()
    
    text = text.split()  
    
    text = [word for word in text if word not in stop_words and len(word)>1]
    
    text = [stemmer.stem(word) for word in text]
    
    text = " ".join(text)
    
    return text

    
def predict_text_by_mlmodel(text,lr_model,text_Encoder):
    text = clean_content(text) #return string
    text = [text] #vectisers not take string as input so make it list
    y_pred = lr_model.predict(text) #lr model having y of label encode so return number in list ex [2]
    # print(y_pred)             #[2]
    emotion = text_Encoder.inverse_transform(y_pred) #corresponding label name , return list ex ['joy']
    # print(emotion)
    print(f"Predicted emotion: {emotion[0]}") 
    return emotion[0]  #return string of emotion name ex 'joy'
    
def predict_text_by_dlmodel(text,dl_model,text_Encoder,tokenizer,MAX_LEN):    
    text = clean_content(text) #return string
    text = [text] #vectisers not take string as input so make it list
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=MAX_LEN, padding='post')
    last_layer_output = dl_model.predict(text)  
    # print(last_layer_output)               #[[0.0716216  0.28323117 0.31504422 0.15884334 0.17125972]] as last layer is softmax return 2d array
    number = np.argmax(last_layer_output,axis=1)  #return 1d array [2]
    emotion = text_Encoder.inverse_transform(number)  #corresponding label name , return list ex ['joy']
    print(f"Predicted emotion: {emotion[0]}")    
    return emotion[0]  #return string of emotion name ex 'joy'
    
    

    
st.write("## Text Emotion Prediction")
    
text_input = st.text_area("Enter text for emotion prediction")

if st.button("Predict Emotion from Text using ML Model"):
    emotion = predict_text_by_mlmodel(text_input, lr_best_model, text_Encoder)
    st.success(f"Predicted Emotion (ML): {emotion}")

if st.button("Predict Emotion from Text using DL Model"):
    emotion = predict_text_by_dlmodel(text_input, grumodel, text_Encoder, tokenizer, MAX_LEN)
    st.success(f"Predicted Emotion (DL): {emotion}")


