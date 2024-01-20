
# import necessary libraries
import pandas as pd
import os
import glob
import numpy as np
import math
import  scipy.stats  
df = pd.read_csv("DEF.csv")
dc = pd.read_csv("Fifa werte.csv")
dg= pd.read_csv("Master.csv")
kl = pd.read_csv("Gesamt.csv")
#print(dg.head(10))

cf= dg.dropna()

print(dg.Position)


boxes = pd.DataFrame(dg, columns=[ "Name", "Rating", "Fifa Ability Overall","Postion"])
#print(boxes.head(50))
box = pd.DataFrame(dc, columns=[ "FIFA", "OPTA", "Fifa Ovr", ""])
print(box.head(10))

#boxes = pd.DataFrame(dc, columns=[ "Name", "Rating", "Fifa Ability Overall"])
#print(boxes.head(10))

print(boxes.set_index("Name", inplace=True))

print(boxes.loc["Cristiano Ronaldo"])

print(box.set_index("FIFA", inplace=True))
print(box.loc["Cristiano Ronaldo"])
'''
#print(df.isnull())
# print(df.fillna(50, inplace = True))
# new = df.dropna()
# print(new.to_string());
# print(new.columns)
# print(new.head(30))
dh= pd.read_csv("Fifa werte.csv", index_col = 0)
print( dh.head(20))
print(dh.columns)


dd= pd.read_csv("Gesamt.csv", index_col = 0)
print( dd.head(20))
print(dd.columns)
'''
'''
def gini_impurity(y):
 
 # Given a Pandas Series, it calculates the Gini Impurity. 
 # y: variable with which calculate Gini Impurity.
  
    
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)
    else:
        raise("object must be pandas series.")
        
print(gini_impurity(dg["Fifa Ability Overall"])) 
       
#print(gini_impurity(dg.Fifa_Ability_Overall))
'''


#table concatenation
kp = pd.concat([dc, dg], axis = 1)
print(kp.columns)
print(kp.head(10)[["Fifa Ovr", "Fifa Ability Overall"]])
#print(s)
 






#Entropy Information Gain for dg
def entropy1(dg, targetcol):
    # store all of our columns and gini scores
    entropy_scores = []
    # iterate through each column in your dataframe
    for col in dg.columns:
       
        if col == targetcol:
           continue
 # get the value_counts normalized, saving us having to iterate through
        # each variable
        Fifa_Ability_Overall = dg[col].value_counts(normalize=True, sort=False)
    
    
    # calculate our entropy for the column
        entropy1 = -(Fifa_Ability_Overall * np.log(Fifa_Ability_Overall) / np.log(math.e)).sum()
        
        print(f'Variable {col} has Entropy of {round(entropy1,4)}\n')
    
    
    # append our column name and gini score
    entropy_scores.append((col,entropy1))
    # sort our gini scores lowest to highest
    split_pairs = sorted(entropy_scores, key=lambda x: -x[1], reverse=True)[0]
    # print out the best score
    print(f'''Split on {split_pairs[0]} With Information Gain of {round(1-split_pairs[1],3)}''')
        
        

final1 = entropy1(dg, 'Fifa')
final1

#Entropy Information Gain for dc
print("-----------------------------------------------------------------------------")
print("\n")
print(" THIS FOLLOWING VALUES IS FOR THE FIFA WERTE TABLE\n")
def entropy(dc, targetcol):
    # store all of our columns and gini scores
    entropy_scores = []
    # iterate through each column in your dataframe
    for col in dc.columns:
       
       if col == targetcol:
           continue
 # get the value_counts normalized, saving us having to iterate through
        # each variable
       Fifa_Ovr= dc[col].value_counts(normalize=True, sort=False)
    
    
    # calculate our entropy for the column
       entropy = -(Fifa_Ovr * np.log(Fifa_Ovr) / np.log(math.e)).sum()
        
       print(f'Variable {col} has Entropy of {round(entropy,4)}\n')
    
    
    # append our column name and gini score
       entropy_scores.append((col,entropy))
    # sort our gini scores lowest to highest
    split_pair = sorted(entropy_scores, key=lambda x: -x[1], reverse=True)[0]
    # print out the best score
    print(f'''Split on {split_pair[0]} With Information Gain of {round(1-split_pair[1],3)}''')
        
        

final = entropy(dc, "Fifa Rating")
final 


  