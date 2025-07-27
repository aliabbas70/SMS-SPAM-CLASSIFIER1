import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = text.split()  # ✅ Replacing nltk.word_tokenize with basic split

    y = []
    for i in text:
        i = ''.join(char for char in i if char.isalnum())  # remove punctuations
        if i and i not in stopwords.words('english'):
            y.append(ps.stem(i))

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output
    if result == 1:
        st.header("❌ Spam")
    else:
        st.header("✅ Not Spam")


def transform_text(text):
    text= text.lower()#to convert to lower case
    text = nltk.word_tokenize(text)# to tokenize

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i) #this logic to remove special characters

    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]   
    y.clear()
    for i in text:
        y.append(ps.stem(i))  
    return " ".join(y)    
            


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms=st.text_area("Enter the message")
if st.button('Predict'):

#1. preprocess
   transformed_sms=transform_text(input_sms)
#2.vectorize
   vector_input=tfidf.transform([transformed_sms])
#3.predict
   result=model.predict(vector_input)[0]
#4.Display 
   if result==1:
    st.header("Spam")
   else:
    st.header("Not Spam")
