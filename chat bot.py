
# coding: utf-8

# In[1]:


import en_core_web_lg # Large SpaCy model for English language
import numpy as np
import re # regular expressions
import spacy # NLU library

from collections import defaultdict
from sklearn.svm import SVC # Support Vector Classification model


# In[2]:


output_format = "IN: {input}\nOUT: {output}\n" + "_"*50


# In[3]:


# Create training data
training_sentences = [
    "hi",
    "hello",
    "hey",
    "Hey there",
    "yo",
    "good morning",
    "morning",
    "gm",
    "good morning!",
    "how are you?",
    "how have you been?",
    "how's everything?",
    "how's it going?",
    "what's going on?",
    "whassup?",
    "what's up",
    "how r u"
    
    
]
training_intents = [
    "hello_pref",
    "hello_pref",
    "hello_pref",
    "hello_pref",
    "hello_pref",
    "morning_wish",
    "morning_wish",
    "morning_wish",
    "morning_wish",
    "how_r_u",
    "how_r_u",
    "how_r_u",
    "how_r_u",
    "how_r_u",
    "how_r_u",
    "how_r_u",
    "how_r_u"
]


# In[4]:


# this may take a couple of seconds
nlp = en_core_web_lg.load()


# In[5]:


# Initialize the array with zeros: X
X_train = np.zeros((len(training_sentences), 
              nlp('sentences').vocab.vectors_length))

for i, sentence in enumerate(training_sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X_train[i, :] = doc.vector


# In[6]:


# Create a support vector classifier
clf = SVC(C=1, gamma="auto", probability=True)

# Fit the classifier using the training data
clf.fit(X_train, training_intents)

#Yes, a lot can be done here to check / improve model performance! We will leave that for another day!


# In[7]:


def get_intent_ml(text):
    doc = nlp(text)
    return(clf.predict([doc.vector])[0])


# In[8]:


responses_ml = {
    "hello_pref":"hello! what can i help you ",
    "how_r_u": "Thanks for asking. I am fine!",
    "default":"sorry i cant understand"
}


# In[9]:


def respond_ml(text):
    response = responses_ml.get(get_intent_ml(text), responses_ml["default"])
    return(output_format.format(input=text, output=response))


# In[10]:


print(respond_ml('hello'))

