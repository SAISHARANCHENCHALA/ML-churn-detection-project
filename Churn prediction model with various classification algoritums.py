
# ## GOAL: Create a model to predict whether or not a customer will Churn .
# 
# ----
# ----
# 
# 
# ## Complete the Tasks in Bold Below!
# 
# ## Part 0: Imports and Read in the Data
# 
# **TASK: Run the filled out cells below to import libraries and read in your data. The data file is "Telco-Customer-Churn.csv"**

# In[132]:


# RUN THESE CELLS TO START THE PROJECT!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[133]:


df = pd.read_csv('../DATA/Telco-Customer-Churn.csv')


# In[134]:


df.head()


# ## Part 1: Quick Data Check
# 
# **TASK: Confirm quickly with .info() methods the datatypes and non-null values in your dataframe.**

# In[135]:


 


# In[136]:


df.info()


# **TASK: Get a quick statistical summary of the numeric columns with .describe() , you should notice that many columns are categorical, meaning you will eventually need to convert them to dummy variables.**

# In[137]:


 


# In[138]:


df.describe()


# # Part 2:  Exploratory Data Analysis
# 
# ## General Feature Exploration
# 
# **TASK: Confirm that there are no NaN cells by displaying NaN values per feature column.**

# In[139]:


 


# In[140]:


df.isna().sum()


# **TASK:Display the balance of the class labels (Churn) with a Count Plot.**

# In[141]:


 


# In[142]:


sns.countplot(data=df,x='Churn')


# <img src='fig1.png' >

# **TASK: Explore the distrbution of TotalCharges between Churn categories with a Box Plot or Violin Plot.**

# In[143]:



# In[144]:


sns.violinplot(data=df,x='Churn',y='TotalCharges')


# <img src='fig2.png' >

# **TASK: Create a boxplot showing the distribution of TotalCharges per Contract type, also add in a hue coloring based on the Churn class.**

# In[145]:


 


# In[146]:


plt.figure(figsize=(10,4),dpi=200)
sns.boxplot(data=df,y='TotalCharges',x='Contract',hue='Churn')
plt.legend(loc=(1.1,0.5))


# <img src='fig3.png' >

# **TASK: Create a bar plot showing the correlation of the following features to the class label. Keep in mind, for the categorical features, you will need to convert them into dummy variables first, as you can only calculate correlation for numeric features.**
# 
#     ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 
#      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService',
#        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
# 
# ***Note, we specifically listed only the features above, you should not check the correlation for every feature, as some features have too many unique instances for such an analysis, such as customerID***

# In[147]:


 


# In[148]:


df.columns


# In[149]:


corr_df  = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod','Churn']]).corr()


# In[150]:


corr_df['Churn_Yes'].sort_values().iloc[1:-1]


# In[151]:


plt.figure(figsize=(10,4),dpi=200)
sns.barplot(x=corr_df['Churn_Yes'].sort_values().iloc[1:-1].index,y=corr_df['Churn_Yes'].sort_values().iloc[1:-1].values)
plt.title("Feature Correlation to Yes Churn")
plt.xticks(rotation=90);


# <img src='figbar.png'>

# ---
# ---
# 
# # Part 3: Churn Analysis
# 
# **This section focuses on segementing customers based on their tenure, creating "cohorts", allowing us to examine differences between customer cohort segments.**

# **TASK: What are the 3 contract types available?**

# In[152]:





# In[153]:


df['Contract'].unique()


# **TASK: Create a histogram displaying the distribution of 'tenure' column, which is the amount of months a customer was or has been on a customer.**

# In[154]:





# In[155]:


plt.figure(figsize=(10,4),dpi=200)
sns.histplot(data=df,x='tenure',bins=60)


# <img src="fig5.png">

# **TASK: Now use the seaborn documentation as a guide to create histograms separated by two additional features, Churn and Contract.**

# In[156]:


 


# In[157]:


plt.figure(figsize=(10,3),dpi=200)
sns.displot(data=df,x='tenure',bins=70,col='Contract',row='Churn');


# <img src="fig6.png">

# **TASK: Display a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn.**

# In[158]:


 


# In[159]:


df.columns


# In[160]:


plt.figure(figsize=(10,4),dpi=200)
sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn', linewidth=0.5,alpha=0.5,palette='Dark2')


# <img src='fig7.png'>

# ### Creating Cohorts based on Tenure
# 
# **Let's begin by treating each unique tenure length, 1 month, 2 month, 3 month...N months as its own cohort.**
# 
# **TASK: Treating each unique tenure group as a cohort, calculate the Churn rate (percentage that had Yes Churn) per cohort. For example, the cohort that has had a tenure of 1 month should have a Churn rate of 61.99%. You should have cohorts 1-72 months with a general trend of the longer the tenure of the cohort, the less of a churn rate. This makes sense as you are less likely to stop service the longer you've had it.**

# In[161]:


 


# In[162]:


no_churn = df.groupby(['Churn','tenure']).count().transpose()['No']
yes_churn = df.groupby(['Churn','tenure']).count().transpose()['Yes']


# In[163]:


churn_rate = 100 * yes_churn / (no_churn+yes_churn)


# In[164]:


churn_rate.transpose()['customerID']


# **TASK: Now that you have Churn Rate per tenure group 1-72 months, create a plot showing churn rate per months of tenure.**

# In[165]:


 


# In[166]:


plt.figure(figsize=(10,4),dpi=200)
churn_rate.iloc[0].plot()
plt.ylabel('Churn Percentage');


# <img src='fig9.png'>

# ### Broader Cohort Groups
# **TASK: Based on the tenure column values, create a new column called Tenure Cohort that creates 4 separate categories:**
#    * '0-12 Months'
#    * '24-48 Months'
#    * '12-24 Months'
#    * 'Over 48 Months'    

# In[167]:





# In[168]:


def cohort(tenure):
    if tenure < 13:
        return '0-12 Months'
    elif tenure < 25:
        return '12-24 Months'
    elif tenure < 49:
        return '24-48 Months'
    else:
        return "Over 48 Months"


# In[169]:


df['Tenure Cohort'] = df['tenure'].apply(cohort)


# In[170]:


df.head(10)[['tenure','Tenure Cohort']]


# **TASK: Create a scatterplot of Total Charges versus Monthly Charts,colored by Tenure Cohort defined in the previous task.**

# In[171]:


 


# In[172]:


plt.figure(figsize=(10,4),dpi=200)
sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Tenure Cohort', linewidth=0.5,alpha=0.5,palette='Dark2')


# <img src='fig10.png'>

# **TASK: Create a count plot showing the churn count per cohort.**

# In[ ]:


 


# In[295]:


plt.figure(figsize=(10,4),dpi=200)
sns.countplot(data=df,x='Tenure Cohort',hue='Churn')


# <img src='cplot.png'>

# **TASK: Create a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and colored by the Churn hue.**

# In[174]:


 


# In[175]:


plt.figure(figsize=(10,4),dpi=200)
sns.catplot(data=df,x='Tenure Cohort',hue='Churn',col='Contract',kind='count')


# <img src='fig11.png'>

# -----
# 
# # Part 4: Predictive Modeling
# 
# **Let's explore 4 different tree based methods: A Single Decision Tree, Random Forest, AdaBoost, Gradient Boosting. Feel free to add any other supervised learning models to your comparisons!**
# 
# 
# ## Single Decision Tree

# **TASK : Separate out the data into X features and Y label. Create dummy variables where necessary and note which features are not useful and should be dropped.**

# In[178]:


 


# In[181]:


X = df.drop(['Churn','customerID'],axis=1)
X = pd.get_dummies(X,drop_first=True)


# In[182]:


y = df['Churn']


# **TASK: Perform a train test split, holding out 10% of the data for testing. We'll use a random_state of 101 in the solutions notebook/video.**

# In[183]:


 


# In[184]:


from sklearn.model_selection import train_test_split


# In[185]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# **TASK: Decision Tree Perfomance. Complete the following tasks:**
#    1. Train a single decision tree model (feel free to grid search for optimal hyperparameters).
#    2. Evaluate performance metrics from decision tree, including classification report and plotting a confusion matrix.
#    2. Calculate feature importances from the decision tree.
#    4. OPTIONAL: Plot your tree, note, the tree could be huge depending on your pruning, so it may crash your notebook if you display it with plot_tree.

# In[222]:


from sklearn.tree import DecisionTreeClassifier


# In[223]:


dt = DecisionTreeClassifier(max_depth=6)


# In[224]:


dt.fit(X_train,y_train)


# In[225]:


preds = dt.predict(X_test)


# In[226]:


from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report


# In[227]:


print(classification_report(y_test,preds))


# In[228]:


plot_confusion_matrix(dt,X_test,y_test)


# In[229]:


imp_feats = pd.DataFrame(data=dt.feature_importances_,index=X.columns,columns=['Feature Importance']).sort_values("Feature Importance")


# In[230]:


plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Feature Importance'),x=imp_feats.sort_values('Feature Importance').index,y='Feature Importance')
plt.xticks(rotation=90)
plt.title("Feature Importance for Decision Tree");


# In[231]:


from sklearn.tree import plot_tree


# In[233]:


plt.figure(figsize=(12,8),dpi=150)
plot_tree(dt,filled=True,feature_names=X.columns);


# <img src='hugetree.png'>

# ## Random Forest
# 
# **TASK: Create a Random Forest model and create a classification report and confusion matrix from its predicted results on the test set.**

# In[259]:


 


# In[260]:


from sklearn.ensemble import RandomForestClassifier


# In[266]:


rf = RandomForestClassifier(n_estimators=100)


# In[267]:


rf.fit(X_train,y_train)


# In[268]:


preds = rf.predict(X_test)


# In[269]:


print(classification_report(y_test,preds))


# In[270]:


plot_confusion_matrix(dt,X_test,y_test)


# ## Boosted Trees
# 
# **TASK: Use AdaBoost or Gradient Boosting to create a model and report back the classification report and plot a confusion matrix for its predicted results**

# In[ ]:


 


# In[288]:


from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier


# In[289]:


ada_model = AdaBoostClassifier()


# In[290]:


ada_model.fit(X_train,y_train)


# In[291]:


preds = ada_model.predict(X_test)


# In[292]:


print(classification_report(y_test,preds))


# In[293]:


plot_confusion_matrix(dt,X_test,y_test)


# **TASK: Analyze your results, which model performed best for you?**

# In[294]:


# With base models, we got best performance from an AdaBoostClassifier, but note, we didn't do any gridsearching AND most models performed about the same on the data set.


# ### Great job!
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('../DATA/Telco-Customer-Churn.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df['Contract'].unique()


# In[6]:


df['PaymentMethod'].unique()


# In[7]:


df['OnlineBackup'].map({'Yes':1,'No':0})


# In[8]:


sns.pairplot(data=df,hue='Churn')


# In[107]:





# In[9]:


dummies_data=pd.get_dummies(df.set_index('customerID'),drop_first=True)


# In[10]:


dummies_data


# In[26]:


sns.countplot(data=df,x=df['Churn'],hue='PaymentMethod')


# In[30]:


dummies_data.corr()['Churn_Yes'].sort_values().plot(kind='bar')


# In[33]:


X=dummies_data.drop('Churn_Yes',axis=1)


# In[34]:


y=dummies_data['Churn_Yes']


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


from sklearn.preprocessing import StandardScaler


# In[54]:


from sklearn.neighbors import KNeighborsClassifier


# In[57]:


from sklearn.linear_model import LogisticRegression


# In[70]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[48]:


scaler=StandardScaler()


# In[51]:


xtrain=scaler.fit_transform(X_train)


# In[52]:


xtest=scaler.transform(X_test)


# In[62]:


error=[]

for n in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(xtrain,y_train)
    ypred=knn.predict(xtest)
    error1=1-accuracy_score(ypred,y_test)
    error.append(error1)


# In[63]:


plt.plot(range(1,30),error)


# In[76]:


knn=KNeighborsClassifier(n_neighbors=24)
   


# In[77]:


knn.fit(xtrain,y_train)


# In[78]:


ypred=knn.predict(xtest)


# In[79]:


error1=1-accuracy_score(ypred,y_test)
error1


# In[80]:


confusion_matrix(y_test,ypred)


# In[81]:


print(classification_report(y_test,ypred))


# In[88]:


log=LogisticRegression(penalty='elasticnet',C=0.1,solver='saga',l1_ratio=0.2)


# In[89]:


log.fit(xtrain,y_train)


# In[95]:


ypred=log.predict(xtest)


# In[96]:


confusion_matrix(y_test,ypred)


# In[97]:


print(classification_report(y_test,ypred))


# In[98]:


from sklearn.svm import SVC


# In[99]:


svc=SVC()


# In[100]:


svc.fit(xtrain,y_train)


# In[103]:


ypred=svc.predict(xtest)


# In[104]:


print(classification_report(y_test,ypred))


# In[ ]: