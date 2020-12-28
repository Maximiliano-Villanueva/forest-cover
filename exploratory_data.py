import os
import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt


def plotCorrMatrix(df):
    with pd.option_context('display.max_columns', None):
        fig, ax = plt.subplots(figsize=(40,30))
        sns.heatmap(df.corr(), annot=True, lineWIdths=0.5, ax = ax)
        
def plotOutliers(train, xcol, ycol):
    fig, ax = plt.subplots()
    ax.scatter(train[xcol], train[ycol])
    plt.ylabel(ycol, fontsize=13)
    plt.xlabel(xcol, fontsize=13)
    plt.show()




def boxPlot(df, columns):
    df.boxplot(columns)
    sns.countplot(x=columns,data=df)
    #df["Cover_Type"].value_counts()
        

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    
    q3 = df[col].quantile(0.75)
    
    iqr = q3 - q1
    
    min_threshold = q1 - 1.5*iqr
    
    max_threshold = q3 + 1.5*iqr
    

    df.loc[df[col]>=max_threshold , col] = df[col].mean()
    
    df.loc[df[col]<=min_threshold , col] = df[col].mean()
    

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
target class seems to be balanced
df.info()
boxPlot(df, "Cover_Type")
"""

"""
displots with outliers
"""

cols_outliers = ['Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
                 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
if False:
    
    ax_rows = 4
    ax_cols = 2
    
    fig, ax = plt.subplots(ax_rows, ax_cols)
    
    ax_row = 0
    ax_col = 0
    
    for c in cols_outliers:
        sns.distplot(df[c], ax = ax[ax_row][ax_col])
        print(df[c].describe() )
        
        ax_col +=1
        
        if ax_col == ax_cols:
            
            ax_row +=1
            ax_col = 0


    
"""
boxplots with outliers
"""
if False: 
    
    ax_cols = len(cols_outliers)
    
    fig, axs = plt.subplots(ncols= ax_cols, figsize=(50,15))

    
    for c_index in range(len(cols_outliers)):
    
        sns.boxplot(df[cols_outliers[c_index]], ax = axs[c_index])
        print(df[cols_outliers[c_index]].describe() )



if False:
    continuous_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    out_label = 'Cover_Type'
    
    for index in range(len(continuous_columns)):
      plt.figure(index, figsize=(10,5))
      sns.boxplot(y = df[continuous_columns[index]], x = df[out_label])



"""
categorical features
"""

cat_columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

out_label = 'Cover_Type'

if False:
    for index in range(len(cat_columns)):
        
        plt.figure(index, figsize=(10,5))
        
        df.groupby(out_label)[cat_columns[index]].value_counts().plot.bar()
        plt.title(cat_columns[index])
        
        plt.show()
        
        plt.figure(index, figsize=(10,5))
        
        df.groupby(out_label)[cat_columns[index]].sum().plot.bar()
        plt.title(cat_columns[index])
        
        plt.show()

"""
drop type 7 and 15 since they shwo no variation in data
"""
for col in df.columns:
    col_counts = len(df[col].value_counts())
    if col_counts < 2:
        df.drop([col],axis=1,inplace=True)
        dfTest.drop([col],axis=1,inplace=True)
        print( col , " -> " , col_counts)
  #print( df_train[col].value_counts() )




"""
check distributions after removing outliers
"""
for c in cols_outliers:
    df[c] = remove_outliers(df, c)


if False:
    
    ax_rows = 4
    ax_cols = 2
    
    fig, ax = plt.subplots(ax_rows, ax_cols)
    
    ax_row = 0
    ax_col = 0
    
    for c in cols_outliers:
        sns.distplot(df[c], ax = ax[ax_row][ax_col])
        print(df[c].describe() )
        
        ax_col +=1
        
        if ax_col == ax_cols:
            
            ax_row +=1
            ax_col = 0
            
plotCorrMatrix(df.corr())
