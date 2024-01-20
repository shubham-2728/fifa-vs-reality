import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import  scipy.stats  
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
#from six import StringIO from IPython.display import Image  
# Read the dataset
dc = pd.read_csv("Fifa werte.csv")

# Drop any rows with missing values
dc = dc.dropna()

# Select relevant columns for features and target
feature_cols = ['OPTA', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'StandingTackle']
target_col = 'Fifa Ovr'
#select independent and dependent variable
X = dc[feature_cols]#independent variables
y = dc[target_col]# dependent which depends on the values of the other columns
print(X)
print(y)
# Perform label encoding for any categorical columns(variables) changes  to Numeric format.
#due to the fact that most maschine Leaning models work best with Numerical values
label_encoder = LabelEncoder()
X_encoded = X.copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X_encoded[col] = label_encoder.fit_transform(X[col])

# Perform one-hot encoding for categorical columns
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(onehot_encoder.fit_transform(X_encoded))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=2)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf=clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)
print(y_pred)
# Evaluate the model accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
#OR THIS METHOD
#to get predicted value not need i just for funny
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
#predicting a new result

y_pred = regressor.predict(X_test)
print(y_pred)
tree.plot_tree(clf)
