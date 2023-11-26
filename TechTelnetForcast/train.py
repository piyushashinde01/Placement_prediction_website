import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np  
df=pd.read_csv(r'D:\2\data.csv ')


df.info() 

df=df.drop(columns='StudentID')
data=np.array(df)
# df.groupby('PlacementStatus').size().plot(kind='pie')
# plt.title('Placement Distribution')

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for columns in object_cols:
    df[columns]=labelencoder.fit_transform(df[columns])

# correlation_matrix=df.corr()
# plt.figure(figsize=(12,6))
# sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')

# sns.countplot(data=df,x='CGPA')
# plt.xticks(rotation=90)
# plt.title('CGPA Analysis')

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='CGPA')
# plt.xticks(rotation=90)
# plt.title('CGPA wise Placement')

# sns.displot(df['HSC_Marks'])

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='HSC_Marks')
# plt.xticks(rotation=90)
# plt.title('HSC Marks wise Placement')

# sns.displot(df['SSC_Marks'])

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='SSC_Marks')
# plt.xticks(rotation=90)
# plt.title('SSC Marks wise Placement')

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='ExtracurricularActivities')
# plt.xticks(rotation=90)
# plt.title('ExtracurricularActivities wise Placement')

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='SoftSkillsRating')
# plt.xticks(rotation=90)
# plt.title('Softskills wise Placement')

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='Projects')
# plt.xticks(rotation=90)
# plt.title('Projects wise Placement')

# sns.countplot(data=df.loc[(df.PlacementStatus==1)],x='AptitudeTestScore')
# plt.xticks(rotation=90)
# plt.title('AptitudeTestScore wise Placement')

# sns.pairplot(df,kind='hist',diag_kind='hist',)

X=df.drop(columns='PlacementStatus')
y=df.PlacementStatus

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
lr=LogisticRegression(max_iter=50000,penalty=None)
lr.fit(X_train,y_train)
# prediction=lr.predict(X_test)
# print(accuracy_score(prediction,y_test))
# target_names=['NotPlaced','Placed']
# print(classification_report(y_test,prediction,target_names=target_names))
# cm=confusion_matrix(y_test,prediction,normalize='true')
# ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names).plot(cmap='Greens')



import pickle
pickle.dump(lr,open('lr.pkl','wb'))

lr=pickle.load(open('lr.pkl','rb'))
#output = lr.predict([[8.6,1,3,2,78,4.1,1,0,86.9,83.1]])
#if output == 1:
  #  print("YES You can Place ")
#else:
 #   print("NO")