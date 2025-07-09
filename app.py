#========================import packages=========================================================
import streamlit as st
import numpy as np
# import nltk
import re
import os
import pickle
# import matplotlib.pyplot as plt
import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# IMAGE_SIZE = 256
# Main_dir = "/Users/somesh/Desktop/IITH_PROJECTS/Human_Emotion_detector"

# image_model = load_model(os.path.join(Main_dir,'models','image_model.h5'))
# IMAGE_CLASS_NAMES = pickle.load(open(os.path.join(Main_dir,'models','IMAGE_CLASS_NAMES.pkl'),'rb'))


# text_Encoder = pickle.load(open(os.path.join(Main_dir,'models','text_Encoder.pkl'),'rb'))
# tokenizer = pickle.load(open(os.path.join(Main_dir,'models','tokenizer.pkl'),'rb'))
# MAX_LEN = pickle.load(open(os.path.join(Main_dir,'models','MAX_LEN.pkl'),'rb'))


# lr_best_model = pickle.load(os.path.join(Main_dir,'models','lr_best_model.pkl'))
# lstmmodel = pickle.load(open(os.path.join(Main_dir,'models','lstmmodel.h5'),'rb'))

# def predict_image_emotion(image_path,model):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
#     image = image/(IMAGE_SIZE-1)
#     plt.imshow(image)
#     # print(f"image shape is {image.shape}")
#     image = np.expand_dims(image,axis=0)
#     # print(f"batch image shape is {image.shape}")
#     layer_output = model.predict(image)  
#     # print(f"layer output shape is {layer_output.shape}")
#     result = np.argmax(layer_output)
#     # print(f"result is {result}")
#     print(f"predicted emotion is {IMAGE_CLASS_NAMES[result]}")
    
    
# def clean_content(text):
    
#     text = re.sub(r'[^a-zA-Z]',' ',text)
    
#     text = re.sub(r'\s+', ' ', text)
    
#     text = text.lower()
    
#     text = text.split()  
    
#     text = [word for word in text if word not in stop_words and len(word)>1]
    
#     text = [stemmer.stem(word) for word in text]
    
#     text = " ".join(text)
    
#     return text

    
# def predict_text_by_mlmodel(text,lr_model,text_Encoder):
#     text = clean_content(text) #return string
#     text = [text] #vectisers not take string as input so make it list
#     y_pred = lr_model.predict(text) #lr model having y of label encode so return number in list ex [2]
#     # print(y_pred)             #[2]
#     emotion = text_Encoder.inverse_transform(y_pred) #corresponding label name , return list ex ['joy']
#     # print(emotion)
#     print(f"Predicted emotion: {emotion[0]}") 
    
# def predict_text_by_dlmodel(text,lstm_model,text_Encoder,tokenizer,MAX_LEN):    
#     text = clean_content(text) #return string
#     text = [text] #vectisers not take string as input so make it list
#     text = tokenizer.texts_to_sequences(text)
#     text = pad_sequences(text, maxlen=MAX_LEN, padding='post', truncating='post')
#     last_layer_output = lstm_model.predict(text)  
#     # print(last_layer_output)               #[[0.0716216  0.28323117 0.31504422 0.15884334 0.17125972]] as last layer is softmax return 2d array
#     number = np.argmax(last_layer_output,axis=1)  #return 1d array [2]
#     emotion = text_Encoder.inverse_transform(number)  #corresponding label name , return list ex ['joy']
#     print(f"Predicted emotion: {emotion[0]}")    
    
    
    
st.title("Human Emotion Detector App")
st.write("This app predicts the emotion of a person from a text or an image")