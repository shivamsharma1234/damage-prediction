
#Libraries

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import pickle
import numpy as np


def readType():

    with open('dataTypes.json', 'r') as f:
        data_types = json.load(f)
        
    with open('TrainDataTypes.json', 'r') as f:
        train_data_types = json.load(f)
        
    data_types.pop('Row_ID')
    data_types.pop('Household_ID')
    data_types.pop('Claim_Amount')

    same_col = []
    cols4dummies = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'Cat8', 'Cat9', 'Cat10', 'Cat11', 'Cat12', 'OrdCat', 'NVCat']
    for i in data_types.keys():#list(data_types.keys())[0:5] + list(data_types.keys())[18:31]
        if i not in cols4dummies:
            same_col.append(i)
    return data_types, train_data_types, same_col





# In[267]:

def loadModel():
    # load the model from disk
    loaded_model = pickle.load(open('RandomForestModel.sav', 'rb'))
    return loaded_model









def encoderLoad():
    pickleLoaded = dict()
    for col in ['Blind_Make','Blind_Model','Blind_Submodel']:
        pkl_file = open(col+'.pkl', 'rb')
        pick = pickle.load(pkl_file) 
        pickleLoaded[col] = pick
        pkl_file.close()
    return pickleLoaded



def predict(test_inpt, data_types, train_data_types, same_col):

    pickleLoaded = encoderLoad()
    loaded_model = loadModel()
    colm = list(train_data_types.keys())
    mod_inputs = pd.DataFrame(columns=colm)
    mod_inputs = mod_inputs.astype(train_data_types) 
    data = []
    
    for col in colm:
        if col in same_col:
            
            if col in ['Blind_Make','Blind_Model','Blind_Submodel']:
                le = pickleLoaded[col]
                test_inpt[col] = le.transform(test_inpt[col])
                data.append(test_inpt.loc[0,col])
            else:
                data.append(test_inpt.loc[0,col])
        else:
            data.append(0)

    s2 = pd.Series(data, index=list(train_data_types.keys()))
    mod_inputs = mod_inputs.append(s2,ignore_index=True)
    result = loaded_model.predict([mod_inputs.iloc[0,:]])
    return result[0]



def testInput(data, data_types):
    test_inpt = pd.DataFrame(columns=data_types.keys())
    test_inpt = test_inpt.astype(data_types) 

    s2 = pd.Series(data, index=list(data_types.keys()))
    test_inpt = test_inpt.append(s2,ignore_index=True)
    return test_inpt

    
def main():
    data_types, train_data_types, same_col = readType()

    df = pd.read_csv("podatki.csv")

    print("Number of data points:",df.shape[0])
    print("Number of data features:",df.shape[1])
    df.drop(['Row_ID','Household_ID','Claim_Amount'], axis=1, inplace=True)

    

    data = []
    for i in range(len(data_types.keys())):
        data.append(df.iloc[10880,i])
    test_inpt = testInput(data, data_types)

    t = predict(test_inpt, data_types, train_data_types, same_col)
    print("Predicted Result is",t)


if __name__ == "__main__":
    main()