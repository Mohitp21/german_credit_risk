#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.German Credit rating data was used which had features related to client’s financial status, background.
#Output was predicting a clientas high risk or low risk.
#2. Implemented a variety of classification models like Logistic Regression, Decision Tree, Naïve Bayes.
#3. Found the best model using confusion matrix, ROC curve techniques.


# In[2]:


#1. Classification Problem are an important category of problems in analytics in which the outcome variable or response variable
#(Y)takes discrete values.
#2.Classification model predict probability of observation belonging to class ,known as class probability.
#3. It may have binary or multiple outcomes.
#4. Several techniques for solving classification problems:
#a.Logistic Regression
#b.Clasification Tree(desicion tree)
#c.discriminant Analysis
#d.Support Vector Machine
#e.Neutral network


# In[3]:


import pandas as pd 


# In[4]:


df=pd.read_csv('german_credit_data.csv')
df


# In[5]:


df.info()


# In[6]:


df.iloc[0:5,1:11]


# In[7]:


df1=pd.get_dummies(df,drop_first=True)
df1


# In[8]:


df1.value_counts('Risk_good')


# In[9]:


# The above output displays that there are 700 good credits and 300 observations of bad credits.


# In[10]:


# For building classification model, the risk column will be used as dependent variable , while remaining columns will be 
#independent variables .We will create a list named x_features and store names of all independent variables .


# In[11]:


x_features=list(df.columns)
x_features.remove('Risk')
x_features


# In[12]:


# ONE HOT ENCODING - There are several categorical features in the data which need to be binary encoded using dummy variables.
# Use pd.get_dummies() to encode categorical features.
# Use drop_first= True for for dropping first category as for n categories there can be n-1 dummy variables.


# In[13]:


encoded_df=pd.get_dummies(df[x_features],drop_first=True)
encoded_df


# In[14]:


encoded_df1=pd.get_dummies(df[x_features])
encoded_df1


# In[15]:


encoded_df[['Housing_own','Housing_rent']].head(10)


# In[16]:


# IN above output if both two variable values are 0 then that type is Housing_free which is dropped using drop_first=True


# In[17]:


import statsmodels.api as sm
Y=df1.Risk_good
X=sm.add_constant(encoded_df)


# In[18]:


from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


# In[19]:


# BUILDING LOGISTIC REGRESSION MODEL-
#1. Logit() method available in statsmodels.api module helps to estimate parameters for logistic regression.
#2.Logit () method estimates model parametrs, accuracy,residual values among other details. 


# In[20]:


logit_model=sm.Logit(Y_train,X_train).fit()
logit_model


# In[21]:


logit_model.summary2()


# In[22]:


# MODEL DIAGNOSTICS- Important to validate logistic regression model to ensure its validity and goodness of fit .Following measures-
#1.WALD'S TEST (chi square test)- used for checking statistical significance of indivisual predictor(feature) variables
#It is equivqlent to t test in MLR model.
#2. LIKELIHOOD RATIO TEST- used for statistical significance of overall model.It is used for variable selection.
#3. PSEUDO R square- measure of goodness of model. It doesnt have same interpretation of R square as in MLR model.


# In[23]:


# The logistic model summary above suggest as per walds test , only 8 features are statistically significant .
#p value for likelihood ratio test less than 0.05 indicates overall model is statistically significant.


# In[24]:


def get_significant_vars(lm):
    #store p values and corrsponding column names in a dataframe
    var_p_vals_df=pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars']=var_p_vals_df.index
    var_p_vals_df.columns=['pvals','vars']
    #Filter the column where p-value is less than 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals<=0.05]['vars'])


# In[25]:


significant_vars=get_significant_vars(logit_model)
significant_vars


# In[26]:


logit_1=sm.Logit(Y_train,sm.add_constant(X_train[significant_vars])).fit()


# In[27]:


logit_1.summary2()


# In[28]:


# The negative sign of the coefficient value indicates as the value of this variable increases , probability of being bad credit
#decreases and vice versa for positive values of coeeficient, where probability of bad credit increases with increase in value.


# In[29]:


# Insights of above model-
#1. Pribability of bad credit increases with increase in age.Younger people tend to pay their credit ontime than older people.


# In[30]:


# Predicting on Test Data-
#1.We will use logit_1 model and significant features to predict the class probability.The predict() method will return the 
#predicted class and probabilities for each observation in test dataset 


# In[31]:


y_pred_df=pd.DataFrame({"actual":Y_test,"predicted_prob":logit_1.predict(sm.add_constant(X_test[significant_vars]))})


# In[32]:


# We can print the predictions of few test samples randomly using the sample method of DataFrame.


# In[33]:


y_pred_df.sample(10,random_state=42)


# In[34]:


# To understand how many  observations the model has classified correctly and how many it has not , a cutt off probability 
#needs to be assumed.Let it be 0.5. Above this probability will be predicted as bad credits and rest all as good credits


# In[35]:


# Now we have to iterate through predicted probability of each  observation using map() and tag observation as bad credit(1)
#if probability value is more than 0.5 or as as good credit (0) for value less than 0.5.


# In[36]:


y_pred_df['predicted']=y_pred_df.predicted_prob.map(lambda x: 1 if x>0.5 else 0)
y_pred_df


# In[37]:


y_pred_df.sample(10,random_state=42)


# In[38]:


# It can be noticed that some classifications are corect and some are wrong. For better understanding we build confusion matrix


# In[39]:


# CREATING CONFUSION MATRIX - 1. In the field of machine learning and specifically the problem of statistical classification,
#a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance
#of an classification algorithm, typically a supervised learning one.
#2.It is formed by checking actual values and predicted values of observation in dataset.


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


from sklearn import metrics
def draw_cm(actual,predicted):
    cm=metrics.confusion_matrix(actual,predicted)
    sns.heatmap(cm,annot=True,fmt='.2f',xticklabels=['Bad Credit','Good Credit'],yticklabels=['Bad Credit','Good Credit'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Fig - Confusion Matrix with cutt off probability of 0.5')


# In[42]:


draw_cm(y_pred_df.actual,y_pred_df.predicted)


# In[43]:


# We can note the following-
#1.TRUE POSITIVE- Left top quadrant represents actual bad credit and is correctly classified as bad credit.
#2. FALSE POSITIVE- Left down quadrant represents actual good credit but incorrectly classified as bad credit.
#3. FALSE NEGATIVE- Rigth top quadrant reprents actual bad credit and is incorrectly classified as good credit.
#4. TRUE NEGATIVE- Right bottom quadrant represents actual good credit is correctly clasified as good credit.


# In[44]:


# MEASURING ACCURACIES- In classification model performance is often measured using concept such as :
#a) Senstivity- Ability of model to correctly classify positives and negatives ( True Positive Rate/ Recall). It is conditional
#probability that the predicted class is positive given that actual class is positive. TP/TP+FN
#b) Specificity- ( True Negative Rate).  It is conditional
#probability that the predicted class is negative given that actual class is negative. TN/TN+FP
#c) Precision -  It is conditional
#probability that the actual value  is positive given that prediction by model is positive. TP/TP+FP
#d) F score - Measure that combines precision and recall ( Harmonic mean btw precision and recall)


# In[45]:


# Classification_report() method in sklearn.metrics gives detailed report of accuracies.


# In[46]:


print(metrics.classification_report(y_pred_df.actual,y_pred_df.predicted))


# In[47]:


# The model is very good at identifying the good credits(Y=0) but not good in identifying bad credits(Y=1). This is the result
#for cutt off probability of 0.5 % . This can be improved by choosing right cutt off probability.


# In[53]:


plt.figure(figsize=(8,6))
plt.title('Fig- Distribution of Predicted probability values by model for both good and bad credits')
sns.distplot(y_pred_df[y_pred_df.actual==0]['predicted_prob'],kde=False,color='Darkorange',label='Good Credit')
sns.distplot(y_pred_df[y_pred_df.actual==1]['predicted_prob'],kde=False,color='black',label='Bad Credit')
plt.legend()
plt.show()


# In[54]:


# We can use above chart to understand how distribution of predicted probabilities for bad credit and good credit look like.
# Larger the overlap btw Predicted probabilities for differnt classes , higher will be misclassification.


# In[55]:


#  RECEIVER OPERATING CHARACTERISTICS(ROC) and AREA UNDER THE CURVE(AUC)-
#1. ROC curve can be used to understand the overall performance of a logistic regression model(in general classification model)
# 2. ROC gives the proportions of such pairs that will be correctly classified.
# 3. Roc curve is plot btw senstivity ( True positive rate) on vrtical axis and 1-specificity(False Positive rate)
# 4. draw_roc() method takes actual classes and predicted probability values and draw roc curve.
# 5.metrics.roc_curve() returns different threshold(cutt offs) values and their corresponding false positve and true positve
#rates. Then these values can be taken and plotted to create ROC curve .
#6. metrics.roc_auc_score() returns area under curve


# In[56]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


# In[60]:




# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_pred_df.actual, y_pred_df.predicted_prob)
auc_score=metrics.roc_auc_score(y_pred_df.actual, y_pred_df.predicted_prob)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr,label='ROC curve(area= %0.2f)'%auc_score)
plt.xlabel('False Positive Rate', color="r")
plt.ylabel('True Positive Rate', color="g")
plt.legend(loc='lower right')
plt.title('ROC Curve')
plt.show()


# In[ ]:


# The line above the diagonal line captures how sesnstivity( True positive) and 1-specificity(false positive) when the cutt off
#probability change. Model with higher AUC is prefered and AUC is frequently used for model selection.
#2. AUC of atleast 0.7 is required for practical application of the model.
#3. AUC greater than 0.9 implies outsatanding model.


# In[63]:


auc_score=metrics.roc_auc_score(y_pred_df.actual, y_pred_df.predicted_prob)
round(float(auc_score),2)


# In[64]:


# FINDING OPTIMAL CLASSIFICATION CUTT-OFF: While using logistic regression model ,one of the decision that a data scientist
#has to make is to choose right classification cutt off probability. The overall accuracy , senstivity and specificity will
#depend on the choosen cutt off probability. The following methods are-
#1.YOUDENS INDEX
#2.COST-BASED APPROACH


# In[65]:


#1. YOUDENS INDEX - It is a classification cutt off probability for which J-STATISTIC function is maximized.
# where J-STATISTIC= MAX [SENSTIVITY + SPECIFICITY -1]


# In[67]:


# IN this we have to select that probability for which ( TPR + TNR-1) or ( TPR - FPR) is maximum. 


# In[72]:


tpr_fpr=pd.DataFrame({'tpr':tpr,'fpr':fpr,'thresholds':thresholds})
tpr_fpr['diff']= tpr_fpr.tpr-tpr_fpr.fpr


# In[76]:


tpr_fpr.sort_values('diff',ascending=False).head(5)


# In[77]:


# From above result , the optimal cutt off probability is 0.73. Thus, now we can classify beyond this predicted probability
#as bad credits and others as good credits.Now use this probabiility for new confusion matrix.


# In[81]:


y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map(lambda x:1 if x>0.73 else 0)
y_pred_df.sample(10, random_state=42)


# In[86]:


draw_cm(y_pred_df.actual,y_pred_df.predicted_new)
plt.title('Fig - Confusion Matrix with cutt off using Yodens index')


# In[87]:


print(metrics.classification_report(y_pred_df.actual,y_pred_df.predicted_new))


# In[ ]:


# With cutt off probability of 0.73 , model is able to classify the bad credits better and 

