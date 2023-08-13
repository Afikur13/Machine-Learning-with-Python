#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


df=pd.read_csv('categorized_resumes.csv')  # Load the dataset


# In[3]:


df


# In[4]:


df.shape  # Check the dataset shape


# In[5]:


df.isnull().sum()


# In[6]:


df.info() # basic info about the dataset


# In[7]:


df['Category'].unique()  # check the unique values


# In[8]:


# Explore the distribution of categories
df['Category'].value_counts()


# In[9]:


# merging the Resume_str and Resume_html
df['text'] = df['Resume_str']+' '+df['Resume_html']


# In[10]:


print(df['text'])


# In[11]:


x=df.drop(['Category'],axis=1) # drop Category column
x


# In[12]:


y=df['Category']  # Assuming y is a column named 'category' from your DataFrame
y


# In[13]:


x = df['text'].values
y = df['Category'].values


# In[14]:


print(x)


# In[15]:


print(y)


# In[16]:


y.shape 


# In[17]:


# Data preprocessing 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
X= vectorizer.transform(x)


# In[18]:


print(X)


# In[19]:


X.toarray()


# In[20]:


from sklearn.model_selection import train_test_split  


# In[21]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3, random_state=1)  # Split the data


# In[22]:


from sklearn.naive_bayes import MultinomialNB


# In[23]:


model=MultinomialNB()  # Initialize the model


# In[24]:


model.fit(xtrain,ytrain)   # Train the model


# In[25]:


model.score(xtest,ytest)  # model Accuracy


# In[26]:


pred=model.predict(xtest)


# In[27]:


pred


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


accuracy_score(ytest,pred)  


# In[30]:


from sklearn.metrics import confusion_matrix


# In[31]:


confusion_matrix(ytest,pred)


# In[32]:


from sklearn.metrics import classification_report


# In[33]:


print(classification_report(ytest,pred))


# In[34]:


from sklearn.svm import SVC  # Support Vector Machine 


# In[35]:


model=SVC()   # Initialize the model


# In[36]:


model_=model.fit(xtrain,ytrain)  # Train the model


# In[37]:


model_.score(xtest,ytest)  # model Accuracy


# In[38]:


pred=model.predict(xtest)


# In[39]:


pred


# In[40]:


from sklearn.metrics import accuracy_score


# In[41]:


accuracy_score(ytest,pred)


# In[42]:


from sklearn.metrics import confusion_matrix


# In[43]:


confusion_matrix(ytest,pred)


# In[44]:


from sklearn.metrics import classification_report


# In[45]:


print(classification_report(ytest,pred))


# In[ ]:




