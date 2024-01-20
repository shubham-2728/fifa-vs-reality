import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.tree import DecisionTreeClassifier    
df= pd.read_csv("Fifa werte.csv")

#Extracting Independent and dependent Variable  
x= df.iloc[:, [2,3]].values  
y= df.iloc[:, 4].values  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
  
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
X_train= st_x.fit_transform(x_train)    
X_test= st_x.transform(x_test)    

#Fitting Decision Tree classifier to the training set  
#"criterion='entropy': Criterion is used to measure the quality of split, which is calculated 
#by information gain given by entropy
#random_state=0": For generating the random states.
from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(X_train, y_train)  

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
max_features=None, max_leaf_nodes=None,
min_impurity_decrease=0.0,# min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0,random_state=0, splitter='best')
#Predicting the test set result  
y_pred= classifier.predict(X_test)  
print(y_pred)


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)
print(cm)

#Visulaizing the trianing set result  
from matplotlib.colors import ListedColormap  
x_set, y_set =X_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Decision Tree Algorithm (Training set)')  
mtp.xlabel('Fifa')  
mtp.ylabel('prediction')  
mtp.legend()  
mtp.show()  
#test dataseet
#Visualization of test set result will be similar to the visualization 
#of the training set except that the training set will be replaced with the test set.
#The above output is completely different from the rest classification models. 
#It has both vertical and horizontal lines that are splitting the dataset according to the Fifa Ovr and estimated salary variable.
#As we can see, the tree is trying to capture each dataset, which is the case of overfitting.
#Visulaizing the test set result  
from matplotlib.colors import ListedColormap 
import numpy as np 
import matplotlib.pyplot as plt
x_set, y_set =X_test, y_test
x1, x2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=1.0),
    np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=1.0)
)

plt.contourf(
    x1, x2,
    classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('yellow', 'green'))
)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        x_set[y_set == j, 0],
        x_set[y_set == j, 1],
        c=ListedColormap(('yellow', 'green'))(i),
        label=j
    )

plt.title('Decision Tree Algorithm (Test set)')
plt.xlabel('Fifa Ovr')
plt.ylabel('Prediction')
plt.legend()
plt.show()
#As we can see in the above image that there are some green data points within the purple region and vice versa. 
#So, these are the incorrect predictions which we have discussed in the confusion matrix.  