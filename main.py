import os
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from scipy.special import boxcox1p

import seaborn as sns
import matplotlib.pyplot as plt


def plotCorrMatrix(df):
    with pd.option_context('display.max_columns', None):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, lineWIdths=0.5, ax = ax)


def saveResult(model, dfTestPredict , ids, output='results.csv'):
    """
    CHOOSE MODEL AND SAVE RESULTS
    """
    
       
    res = model.predict(dfTestPredict).astype(int)
    
    pdResult = pd.DataFrame(list(zip(ids, res)), columns=['Id', 'Cover_Type'])
    
    #, float_format='%.15f'
    pdResult.to_csv(output, index=False)
    print("saved {} rows".format(pdResult.shape[0]))
    
    
# Function responsible for checking our model's performance on the test data
def testSetResultsClassifier(classifier, x_test, y_test, model_name =''):
    
    predictions = classifier.predict(x_test)
    
    results = []

    accuracy = accuracy_score(y_test, predictions)

    results.append(accuracy)
    
    print("\n\n#---------------- Test set results ({}) ----------------#\n".format(model_name))
    print("Accuracy:")
    print(results)
    
    return results


        

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    
    q3 = df[col].quantile(0.75)
    
    iqr = q3 - q1
    
    min_threshold = q1 - 1.5*iqr
    
    max_threshold = q3 + 1.5*iqr
    
    df.loc[df[col]>=max_threshold , col] = df[col].mean()
    
    df.loc[df[col]<=min_threshold , col] = df[col].mean()
    
    return df


def mark_outliers(df, col):
    q1 = df[col].quantile(0.25)
    
    q3 = df[col].quantile(0.75)
    
    iqr = q3 - q1
    
    min_threshold = q1 - 1.5*iqr
    
    max_threshold = q3 + 1.5*iqr
    
    
    df[col+'_outlier_lower'] = df[col] < min_threshold
        
    df[col+'_outlier_lower'] = df[col] > max_threshold
    
    
    return df
    
          

np.set_printoptions(suppress=True)
"""
LOAD DATASET
"""
pathTrain = os.path.join('data', 'train.csv')
pathTest = os.path.join('data', 'test.csv')
df = pd.read_csv(pathTrain)
dfTest = pd.read_csv(pathTest)




"""
categorical features
"""
cat_columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

cols_outliers = ['Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
                 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']




"""
store ids in a variable and remove them from the dataset
"""

test_id=dfTest['Id']


#df = df.drop(columns=['Soil_Type7', 'Soil_Type15', 'Id'])
df = df.drop(columns=['Id'])

dfTest = dfTest.drop(columns=['Id'])

"""
remove outliers
"""

df_size = df.shape[0]


all_data = pd.concat((df, dfTest)).reset_index(drop=True)

for c in cols_outliers:
    all_data = mark_outliers(all_data, c)

    
all_data['Horizontal_sum'] = (all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Roadways'] + all_data['Horizontal_Distance_To_Fire_Points'])


"""
skew variables
"""


all_data['Horizontal_Distance_To_Hydrology'] = boxcox1p(all_data['Horizontal_Distance_To_Hydrology'], 0.3)

all_data['Hillshade_9am'] = boxcox1p(all_data['Hillshade_9am'], 4.5)
all_data['Hillshade_Noon'] = boxcox1p(all_data['Hillshade_Noon'], 6)


all_data['Soil_Type12_32'] = all_data['Soil_Type32'] + all_data['Soil_Type12']
all_data['Soil_Type23_22_32_33'] = all_data['Soil_Type23'] + all_data['Soil_Type22'] + all_data['Soil_Type32'] + all_data['Soil_Type33']

all_data['binned_elevation'] = [math.floor(v/50.0) for v in all_data['Elevation']]

all_data['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in all_data['Horizontal_Distance_To_Roadways']]



df = all_data[:df_size]
dfTest = all_data[df_size:]





"""
scale features
"""
X = df.loc[:,df.columns != 'Cover_Type']
y = df["Cover_Type"].values



dfTest = dfTest.drop(columns=['Cover_Type'])


"""
build models
"""

rf = RandomForestClassifier(n_estimators = 200)



ada = AdaBoostClassifier(base_estimator=rf,
                         n_estimators=200, 
                         learning_rate=0.3,
                         random_state=1)




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)


ada = ada.fit(X_train, y_train)



rf = rf.fit(X_train, y_train)



testSetResultsClassifier(rf, X_test, y_test, model_name ='Random Forest')
testSetResultsClassifier(ada, X_test, y_test, model_name ='Adaboost ')


saveResult(rf, dfTest , test_id, 'rf.csv')
saveResult(ada, dfTest , test_id, 'ada.csv')
