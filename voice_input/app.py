
import nltk
import numpy as np
from flask import Flask,render_template,request
from keras.models import load_model
# Downloading the model for tokenize message.
nltk.download('punkt')
# Downloads stopwords 
nltk.download('stopwords')
#download wordnet, which contains all lemmas of english language
nltk.download('wordnet')


from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import WordNetLemmatizer

model = load_model("/Users/anaghaanilhajare/Desktop/Projects_folder/final_project/voice_input/model.h5")
response_array = []
# stop_words
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_response", methods =["GET", "POST"])
def predict_response(): 
    if request.method == "POST":
        while True:
            # query = takeCommand()
            # get message from user
            message = request.form['userInput']
            # print('You: ',messg=)
            message = message.lower()
            #     print(query)
            if 'time' in message:
                response = time()
                break
            elif 'date' in message:
                response = date()
                break
            elif 'offline' in message:
                response = quit()
                break
            else:
                # predict intent tag using trained neural network
                tag = predict_intent_tag(message)
                # get complete intent from intent tag
                intent = get_intent(tag)
                # generate random response from intent
                response = random.choice(intent['responses'])
                break
        return render_template('index.html', response_array = response)
    







def clean_corpus(corpus):
  # lowering every word in text
  corpus = [ doc.lower() for doc in corpus]
  cleaned_corpus = []


  stop_words = stopwords.words('english')
  wordnet_lemmatizer = WordNetLemmatizer()
  # iterating over every text
  for doc in corpus:
    # tokenizing text
    tokens = word_tokenize(doc)
    cleaned_sentence = []
    for token in tokens:
      # removing stopwords, and punctuation
      if token not in stop_words and token.isalpha():
        # applying lemmatization
        cleaned_sentence.append(wordnet_lemmatizer.lemmatize(token))
    cleaned_corpus.append(' '.join(cleaned_sentence))
  return cleaned_corpus



import json
with open("intents.json","r") as file:
  intents = json.load(file)



corpus = []
tags = []
for intent in intents['intents']:
  # taking all patterns in intents to train a neural network
  for pattern in intent['patterns']:
    corpus.append(pattern)
    tags.append(intent['tag'])





cleaned_corpus = clean_corpus(corpus)
cleaned_corpus





from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_corpus)





from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(tags).reshape(-1,1))







# if prediction for every tag is low, then we want to classify that
INTENT_NOT_FOUND_THRESHOLD = 0.40
def predict_intent_tag(message):
  message = clean_corpus([message])
  X_test = vectorizer.transform(message)
  y = model.predict(X_test.toarray())
  # if probability of all intent is low, classify it as noanswer
  if y.max() < INTENT_NOT_FOUND_THRESHOLD:
    return 'noanswer'
  prediction = np.zeros_like(y[0])
  prediction[y.argmax()] = 1
  tag = encoder.inverse_transform([prediction])[0][0]
  return tag
print(predict_intent_tag('How you could help me?'))
print(predict_intent_tag('chat bot'))
print(predict_intent_tag('Where\'s my order'))





import random
import time
def get_intent(tag):
  # to return complete intent from intent tag
  for intent in intents['intents']:
    if intent['tag'] == tag:
      return intent



if __name__ == "__main__":
    app.run(debug=True, port=5690, host="0.0.0.0")
    

