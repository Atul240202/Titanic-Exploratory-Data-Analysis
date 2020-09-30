#!/usr/bin/env python
# coding: utf-8

# ## Titanic_EDA_Student Atul Jha

# In[85]:


import numpy as np  #You have to install this library using your compiler terminal "pip install numpy"
import pandas as pd  #You have to install this library using your compiler terminal "pip install pandas"
import matplotlib.pyplot as plt #You have to install this library using your compiler terminal "pip install MATPLOTLIB"
import seaborn as sns #You have to install this library using your compiler terminal "pip install SEABORN"
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.random import randint 


# In[86]:


titanic = pd.read_excel('titanic.xlsx')


# In[87]:


titanic.head()


# ## Variable Identification

# In[88]:


titanic.dtypes


# In[89]:


titanic.shape


# In[90]:


titanic.isnull().sum()


# In[91]:


titanic.nunique()


# In[92]:


titanic.count()


# In[93]:


titanic.describe()


# ## Univariate analysis
# 

# In[94]:


titanic['Survived'].value_counts()


# In[95]:


sns.countplot('Survived',data=titanic)


# In[96]:


sns.countplot('Pclass',data=titanic)


# In[97]:


sns.countplot('Sex',data=titanic)


# In[98]:


sns.countplot('SibSp',data=titanic)


# In[99]:


sns.countplot('Parch',data=titanic)


# In[100]:


sns.countplot('Embarked',data=titanic)


# In[101]:


sns.distplot(titanic['Fare'])


# In[102]:


sns.distplot(titanic['Age'])


# # Bivariate 

# In[103]:


Male_survival = titanic[(titanic['Survived'] == 1) & (titanic['Sex'] == 'male')]
Male_survival


# In[104]:


Female_survival = titanic[(titanic['Survived'] == 1) & (titanic['Sex'] == 'female')]
Female_survival


# ### From the above detail we can conclude that 233 female and 109 male survived. So Female had more probability to survive

# In[105]:


titanic.groupby('Pclass')['Survived'].mean().plot(kind='bar')


# In[106]:


titanic.groupby('SibSp')['Survived'].mean().plot(kind='bar')


# In[107]:


titanic.groupby('Parch')['Survived'].mean().plot(kind='bar')


# In[108]:


titanic.groupby('Sex')['Survived'].mean().plot(kind='bar')


# In[109]:


titanic.groupby('Embarked')['Survived'].mean().plot(kind='bar')


# In[110]:


sns.boxplot(x="Survived", y="Age", data=titanic)


# In[111]:


sns.boxplot(x="Survived", y="Fare", data=titanic)


# In[112]:


sns.jointplot(x="Age", y="Fare", data=titanic)


# # Missing Value Treatment (Not important) As we cannot fill null values of cabin we can just drop it and that can effect my data result

# In[113]:


titanic.dropna()


# In[114]:


titanic['Age']= titanic['Age'].fillna(value= titanic['Age'].mean())
titanic


# # Variable Creation

# In[115]:


titanic['FamilyCount']=titanic.SibSp+titanic.Parch
titanic.head(10)


# In[116]:


del titanic['Cabin_First']


# In[ ]:


titanic['Cabin_Category']=titanic.Cabin.str[0]
titanic.head(10)


# # Variable Tranformation

# Assigning mean value to fare '0'

# In[ ]:


print((titanic.Fare == 0).sum())


# In[ ]:


titanic.Fare = titanic.Fare.replace(0, np.NaN)
titanic[titanic.Fare.isnull()].index


# In[ ]:


titanic.Fare.mean()
titanic.Fare.fillna(titanic.Fare.mean(),inplace=True)
titanic[titanic.Fare.isnull()].index


# In[ ]:


print((titanic.Age == 0).sum())


# In[ ]:


titanic.Age.fillna(titanic.Age.mean(),inplace=True)


# In[ ]:


titanic[titanic.Age.isnull()]


# # I have covered all the steps given by Ankur sir in EDA topic.

# In[117]:


#Major project from here


# In[ ]:





# #Major project from here 

# In[147]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[148]:


titanic = pd.read_excel('titanic.xlsx')


# In[149]:


titanic.head()


# In[150]:


titanic.shape


# In[151]:


titanic.info()


# In[152]:


titanic.describe()


# In[153]:


plt.figure(figsize=(10,5))
sns.pairplot(titanic)


# In[154]:


plt.figure(figsize=(10,5))
titanic.boxplot()


# In[162]:


titanic.head()


# Removing cabin as it has lots of null value

# In[164]:


drop_column = ['Cabin']
titanic.drop(drop_column, axis=1)


# Filling missing values

# In[165]:


titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace = True)


# In[168]:


titanic['Fare'].fillna(titanic['Fare'].median(), inplace = True)


# In[169]:


titanic['Age'].fillna(titanic['Age'].median(), inplace = True)


# In[171]:


print(titanic.isnull().sum())


# Creating Dummies

# In[180]:


all_data = [titanic]


# In[181]:


for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[182]:


titanic.head()


# In[183]:


import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[184]:


for dataset in all_data:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])


# In[185]:


for dataset in all_data:
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare','Average_fare','high_fare'])


# In[186]:


Titanic=titanic


# In[187]:


all_dat=[Titanic]


# In[188]:


for dataset in all_dat:
    drop_column = ['Age','Fare','Name','Ticket']
    dataset.drop(drop_column, axis=1, inplace = True)


# In[190]:


drop_column = ['PassengerId']
Titanic.drop(drop_column, axis=1, inplace = True)


# In[192]:


Titanic.head(2)


# In[193]:


Titanic = pd.get_dummies(Titanic, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])


# In[194]:


Titanic.head()


# # Determining X AND Y

# In[199]:


cols = Titanic.columns
cols=['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'Sex_female', 'Sex_male', 'Title_Master', 'Title_Mr', 'Age_type_Teenage',
     'Age_type_Adult', 'Age_type_Elder', 'Em_type_C', 'Em_type_Q', 'Em_type_S', 'Fare_type_Low_fare', 'Fare_type_median_fare', 
     'Fare_type_Average_fare', 'Fare_type_high_fare']


# In[200]:


Titanic = Titanic[cols]
Titanic


# In[201]:


X = Titanic.iloc[:,:-1].values
y = Titanic.iloc[:,-1].values


# In[202]:


X


# In[203]:


X.shape


# In[204]:


y.shape


# # Test and Train division

# In[205]:


from sklearn.model_selection import train_test_split


# In[206]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.20,random_state=0)


# In[207]:


X_train.shape


# In[208]:


X_test.shape


# In[209]:


y_train.shape


# In[210]:


y_test.shape


# # Linear Regression

# In[211]:


from sklearn.linear_model import LinearRegression


# In[212]:


lm = LinearRegression()


# In[213]:


lm.fit(X_train,y_train)


# In[214]:


print(lm.intercept_)


# In[215]:


Titanic.head()


# In[216]:


print(lm.coef_)


# In[219]:


coeff_df = pd.DataFrame(lm.coef_,pd.DataFrame(X_test, columns=['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'Sex_female', 'Sex_male', 'Title_Master', 'Title_Mr', 'Age_type_Teenage',
     'Age_type_Adult', 'Age_type_Elder', 'Em_type_C', 'Em_type_Q', 'Em_type_S', 'Fare_type_Low_fare', 'Fare_type_median_fare', 
     'Fare_type_Average_fare']).columns,columns=['Coefficient'])
coeff_df


# # Predictions

# In[220]:


y_pred = lm.predict(X_test)


# In[221]:


y_pred


# In[222]:


y_test


# # Model Performance Metrics

# In[223]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[224]:


from math import sqrt
rmse = sqrt(mse)


# In[225]:


print('Mean_Squared_Error :' ,mse)
print('Root_Mean_Squared_Error :' ,rmse)
print('r_square_value :',r_squared)


# In[226]:


AdjustedR= (1 - ((1-r_squared)*199)/(200-6-1))
AdjustedR


# In[227]:


X_test[0]


# In[229]:


lm.predict([[1, 5, 0, 6, 1, 2, 1, 6, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]])


# # Logistic Regression

# In[248]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[249]:


y_pred = lr.predict(X_test)


# In[250]:


y_test


# In[251]:


y_pred


# Evaluation of model using Confusion Matrix

# In[252]:


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)


# In[253]:


TN = confusion [0,0]
FP = confusion [0,1]
FN = confusion [1,0]
TP = confusion [1,1]


# In[254]:


print(confusion)
print ("TN: ", TN)
print ("FP: ", FP)
print ("FN: ", FN)
print ("TP: ", TP)


# In[255]:


confusion_matrix = pd.DataFrame(confusion)
confusion_matrix.columns = ['Predicted No Subscription', 'Predicted Yes Subscription']
confusion_matrix = confusion_matrix.rename(index = {0 : 'Actual No Subscription', 1 : 'Actual Yes Subscription'})
confusion_matrix


# Evaluation metrics for confusion matrix

# In[256]:


from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy1 = (TN+TP)/(TN+TP+FN+FP)
print ("Accuracy from metrics: ", accuracy)
print ("Accuracy Calculated: ", accuracy1)


# In[257]:


print ((FP+FN)/float(TP+TN+FP+FN))
print (round(1-metrics.accuracy_score(y_test, y_pred),4))


# In[258]:


print ("RECALL:", metrics.recall_score(y_test,y_pred))
print("CALCULATED RECALL: ", (TP)/(TP+FN))


# In[259]:


print ("SPECIFICITY/TRUE NEGATIVE RATE:", (TN)/(TN+FP))


# In[260]:


print("FALSE POSITIVE RATE: ",(FN)/(FN+TP))


# In[261]:


print("FALSE NEGATIVE RATE: ",(FP)/(TN+FP))


# In[262]:


print ("Precision: ", round(metrics.precision_score(y_test,y_pred),2))
print ("PRECISION CALCULATED: ", round(TP/float(TP+FP),2))


# In[263]:


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print('Accuracy is  :' ,round(accuracy,2)*100)
print('F1 score is :' ,round(f1,2)*100)
print('Precision is  :',round(precision,2)*100)
print('Recall is  :',round(recall,4)*100)
print('Roc Auc is  :',round(roc_auc,2)*100)


# In[264]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Knn 

# In[231]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[232]:


y_pred = knn.predict(X_test)


# In[233]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test, y_pred)


# In[234]:


accuracy_score(y_test, y_pred)


# In[241]:


error = []
accuracy = []

# Calculating error for K values between 1 and 700
for i in range(1,700,35):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    accuracy.append(accuracy_score(y_test, pred_i))


# In[242]:


plt.figure(figsize=(12, 6))
plt.plot(range(1,700,35), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[243]:


plt.figure(figsize=(12, 6))
plt.plot(range(1,700,35), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy Score K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')


# In[244]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


# In[245]:


y_pred1 = knn.predict(X_test)


# In[246]:


confusion_matrix(y_test, y_pred1)


# In[247]:


accuracy_score(y_test, y_pred1)


# In[ ]:




