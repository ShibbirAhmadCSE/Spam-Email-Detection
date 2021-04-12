#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0 = ham (good), 1 = spam (bad)

#import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#load dataset

df=pd.read_csv('D:\\Data\\emails.csv')
df


# In[3]:


#count values of a column

df['email_type'].value_counts()


# In[4]:


#drop duplicate data

df.drop_duplicates(inplace = True)    


# In[5]:


df


# In[6]:


#check null value

df.isnull().sum()  


# In[7]:


df[df['email'].isnull()]

df[df['email_type'].isnull()]


# In[8]:


#drop rows containing null values 

new_df = df.dropna()


# In[9]:


new_df.isnull().sum()


# In[10]:


new_df


# In[11]:


new_df.email_type.value_counts()


# In[12]:


plt.figure(figsize=(12,6))
sns.countplot(x=new_df['email_type'], label = 'Count')
plt.show()


# In[13]:


#separate x and y

x = new_df.email.values    #independent variable

y = new_df.email_type.values    #dependent variable


# In[14]:


#split dataset into train and test

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


# In[15]:


#preprocessing (feature extraction)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000)
x_train=cv.fit_transform(xtrain)
x_train.toarray()


# In[16]:


x_test = cv.transform(xtest)

x_test.toarray()


# In[17]:


#naive bayes 

from sklearn.naive_bayes import MultinomialNB

model1 = MultinomialNB()
model1.fit(x_train,ytrain)


# In[18]:


#support vector machine 

from sklearn.svm import LinearSVC

model2 = LinearSVC(dual=False)
model2.fit(x_train,ytrain)


# In[19]:


from sklearn.linear_model import LogisticRegression

model3 = LogisticRegression(solver='lbfgs', max_iter=400)
model3.fit(x_train,ytrain)


# In[20]:


#performance matrix
pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)
pred3 = model3.predict(x_test)


# In[21]:


pred1


# In[22]:


pred2


# In[23]:


pred3


# In[24]:


#confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pred1)


# In[25]:


confusion_matrix(ytest,pred2)


# In[26]:


confusion_matrix(ytest,pred3)


# In[27]:


#accuracy

from sklearn.metrics import accuracy_score

accuracy_score(ytest,pred1)   #naive bayes


# In[28]:


accuracy_score(ytest,pred2)   #svm


# In[29]:


accuracy_score(ytest,pred3)  #logisticRegression


# In[30]:


#prediction on random email 

emails = ['hey i am lokking for machine learning tutorial in begali language','hey you win an iphone x giveaway for free please do the survey']

cv_emails = cv.transform(emails)


# In[31]:


model1.predict(cv_emails)


# In[32]:


model2.predict(cv_emails)


# In[33]:


model3.predict(cv_emails)


# In[ ]:




