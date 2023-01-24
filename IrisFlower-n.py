import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
dataframe=pd.read_csv("Iris.csv")
print(dataframe)
dataframe=dataframe.drop('Id',axis=1)
print(dataframe.describe())
print(dataframe.info)
print(dataframe.isnull().sum())


X=dataframe.iloc[:,:-1]
Y=dataframe.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.6,test_size=0.4,random_state=0)
GaussNB=GaussianNB()
GaussNB.fit(X_train,Y_train)
GaussNB_predict=GaussNB.predict(X_test)
Dtree=DecisionTreeClassifier(random_state=0)
Dtree.fit(X_train,Y_train)
Dtree_predict=Dtree.predict(X_test)
print(Dtree_predict)
svm_clf=svm.SVC(kernel="linear")
svm_clf.fit(X_train,Y_train)
svm_clf_predict=svm_clf.predict(X_test)
print(accuracy_score(Y_test,svm_clf_predict))

# Data visualization
sns.pairplot(data=dataframe,kind='scatter')
plt.show()
sns.histplot(x="SepalLengthCm", hue="Species", data=dataframe)
plt.show()
sns.histplot(x="SepalWidthCm", hue="Species", data=dataframe)
plt.show()
sns.histplot(x="PetalLengthCm", hue="Species", data=dataframe)
plt.show()
sns.histplot(x="PetalWidthCm", hue="Species", data=dataframe)
plt.show()