#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


df = pd.read_csv('iris.csv')


# In[20]:


df.head()


# In[21]:


df.tail()


# In[22]:


df.info()


# In[26]:


from sklearn.datasets import load_iris


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[11]:


iris = load_iris()
X = iris.data
y = iris.target


# In[12]:


df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[15]:


y_pred = model.predict(X_test)


# In[16]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[17]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

