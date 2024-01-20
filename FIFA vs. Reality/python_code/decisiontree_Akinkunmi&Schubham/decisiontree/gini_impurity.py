import pandas as pd
import seaborn as sb
import statistics as st
import matplotlib.pyplot as plt
import numpy as np
import os # for reading all
import glob # for reading all
# how to read all the datas at once from the folder
path = os.getcwd()
csv_file = glob.glob(os.path.join(path, "*.csv"))
for f in csv_file:
    df= pd.read_csv

#df= pd.read_csv("Fifa werte.csv")
# allows to see all columns
df= pd.read_csv("Fifa werte.csv", index_col = 0)
#c = df.head(20)
#print(c)
df["good"] = (df.SlidingTackle >= 50).astype("int")

df.drop("SlidingTackle", axis= 1, inplace = True)
print(df)
#funtion for entropy
print(df.columns)
"""
def gini_impurity(y):
 
 # Given a Pandas Series, it calculates the Gini Impurity. 
 # y: variable with which calculate Gini Impurity.
  
    
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)
    else:
        raise("object must be pandas series.")
print(gini_impurity(df.good))


def gini_impurity(y):
 
 # Given a Pandas Series, it calculates the Gini Impurity. 
 # y: variable with which calculate Gini Impurity.
  
    
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)
    else:
        raise("object must be pandas series.")
print(gini_impurity(df.Crossing))

#print(df.rename(columns={"Fifa Ovr", "FifaOvr"}, inplace=True))
""" 
def gini_impurity(y):
 
 # Given a Pandas Series, it calculates the Gini Impurity. 
 # y: variable with which calculate Gini Impurity.
  
  
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)
    else:
        raise("object must be pandas series.")
        
print(" the value is FiFA OVR:",1-gini_impurity(df["Fifa Ovr"]))
print("crossing:",1-gini_impurity(df.Crossing))
print("Finishing:",1-gini_impurity(df.Finishing))
print("ShortPassing:",1-gini_impurity(df.ShortPassing))
print("ShortPassing:",1-gini_impurity(df.Dribbling))
print("LongPassing:",1-gini_impurity(df.LongPassing))
print("StandingTackle:",1-gini_impurity(df.StandingTackle))
print("SlidingTackle:",1-gini_impurity(df.good))

#print(df.rename(columns={ "Fifa Ovr": "FifaOvr"}, inplace= True))
#print(c.to_string)
#print(df.head(20))
#print(df.isnull())
# so do the scaterplot
'''
ax=sb.stripplot(c.OPTA, c.StandingTackle)
ax.set(xlabel = "FifaOvr", ylabel= "FifaOvr")
plt.title("Fifa rating GRAPH")
plt.show()
print(c)
dd = pd.read_csv("pivot.csv")
dc = pd.read_csv("Gesamt.csv")
db = pd.read_csv("Master.csv")
print(st.stdev(df.Crossing))
dg= pd.read_csv("MID.csv")
dm = pd.read_csv("DEF.csv")
dj = pd.read_csv("OFF.csv")
dw = pd.read_csv("Back up.csv")

print(pd.DataFrame(df))
print(df)
pd.set_option("display.max_colwidth", 3000)
print(df.head(40))

#to see all the columns


pd.set_option("display.max_columns",3)
pd.set_option("display.max_row", None)
print(df.head(200))
#see the column names
print(df.columns.tolist())
#check if any value is equals to zero
new = dw.dropna()
print(new.to_string)
print(dw.to_string)
print(df.isnull())
print(df.to_string)
print(df.head(10))
print(df.columns)
print(dd)
print(dc)
print(dc.head(10))
print(df.columns)
print(db)
print(db.columns)
print(dg)
print(dg.columns)
print(df.FIFA,df.OPTA)
print(df.to_string())
print(dw.to_string())
print(dw.Name)
#----------------------------------
#Data clean
#Bad data could be

# -> Empty cells
# -> Data in wrong format
# -> Wrong data
# -> Duplicates
# One way to deal with empty cells is to remove rows that contain empty cells.
new = c.dropna()
print(new.to_string());
print(df.to_string())
print(df)
c = pd.DataFrame(df)
print(c)
#replace empty values with new values can be used to deal with empty cell;
print(df.fillna(50, inplace = True))

#replace for specific columns  with new values can be used to deal with empty cell;
df["FIFA"].fillna(50,inplace= True)

#one way to fix wrong values is to replace them with something else whic is right

df.loc(7, "FIFA")= "ISAAC" # 7 is the row number



#removing rows

for x in df.index:
    if df.loc[x, " FIFA"]>70:
        df.drop(x,inplace=True)
       
       
remove duplicates
df.drop_duplicates(inplace= True)
#to visualize out dataframe
new.plot()
plt.show()
'''
#with scattered plot
