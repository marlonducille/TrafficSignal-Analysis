import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


#python installed here - C:\Users\Karen\AppData\Local\Programs\Python\Python36-32\Scripts
# install modules by : cd ... then pip install [module name]
def process_traffic_count_cartaxi(df):
    #traffic count are in thousands
    cut_points = [0,10000,20000,30000,40000,60000]
    label_names = ["<10000", "10000-20000", "20000-30000", "30000-40000", "40000-60000"]
    df["CarsTaxis_categories"] = pd.cut(df["CarsTaxis"], cut_points, labels=label_names)
    return df

def process_year(df):
    cut_points = [1995,2000,2004,2008,2012,2017]
    label_names = ["1995-2000","2000-2004", "2004-2008", "2008-2012", "2012-2017"]
    df["Year_categories"] = pd.cut(df["Year"], cut_points, labels=label_names)
    return df

def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


def create_holdout_data():
    holdout = pd.read_csv('Trafford_traffic.csv')
    holdout = holdout.sample(n=100)

    link_length = holdout['LinkLength_miles']
    holdout['LinkLength_miles_scaled'] = minmax_scale(link_length)

    # prepare the RoadCategory independent variable
    holdout = create_dummies(holdout, 'RoadCategory')

    # prepare YearCategory independent variable
    holdout = process_year(holdout)
    holdout = create_dummies(holdout, 'Year_categories')

    # prepare the CarsTaxi traffic volume dependent variable
    holdout = process_traffic_count_cartaxi(holdout)
    
    encoder = LabelEncoder()
    holdout['CarsTaxis_categories'] = encoder.fit_transform(holdout['CarsTaxis_categories'])
    

    return holdout


'''
Fields
-------
CP (count point) - unique reference for the road link
CarsTaxis - Traffic volume (in thousands od vehicle miles) for Cars and Taxis

https://www.dft.gov.uk/traffic-counts/cp.php?la=Trafford
'''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''' DATA PREPROCESSING 

'''''''''''''''''''''''''''''''''''''''''''''''''''''''

df = pd.read_csv('Trafford_traffic.csv')

# prepare the LinkLength_miles (the journey distance between the start and end junction) independent variable, 
link_length = df['LinkLength_miles']
df['LinkLength_miles_scaled'] = minmax_scale(link_length)

# prepare the RoadCategory independent variable
df = create_dummies(df, 'RoadCategory')

# prepare YearCategory independent variable
df = process_year(df)
df = create_dummies(df, 'Year_categories')

# prepare the CarsTaxi traffic volume dependent variable
df = process_traffic_count_cartaxi(df)

encoder = LabelEncoder()
df['CarsTaxis_categories'] = encoder.fit_transform(df['CarsTaxis_categories'])
                              
                                                   
columns_X = ['LinkLength_miles_scaled', 'RoadCategory_PM', 'RoadCategory_PR', 'RoadCategory_PU', 'RoadCategory_TM', 'Year_categories_1995-2000', 'Year_categories_2000-2004', 'Year_categories_2004-2008',
      'Year_categories_2008-2012', 'Year_categories_2012-2017']


columns_y =  'CarsTaxis_categories' #['CarsTaxis_categories_40000-60000']


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''' USE LOGISTIC REGRESSION TO FIND THE BEST FEATURES 

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


#calculate accuracy of the model - Logistic Regression
lr = LogisticRegression()
scores = cross_val_score(lr,df[columns_X],df[columns_y],cv=10)
accuracy = scores.mean()


#feature selection - find the most to least important feature
train_X, test_X, train_y, test_y = train_test_split(df[columns_X], df[columns_y], test_size=0.2, random_state=0)
lr.fit(train_X, train_y)
coefficients = lr.coef_
feature_importance = pd.Series(coefficients[0],index=train_X.columns).abs().sort_values()


#after finding the best features, column_X now becomes
                                                   
columns_X = ['LinkLength_miles_scaled', 'RoadCategory_PR', 'RoadCategory_TM', 'Year_categories_2004-2008',
       'Year_categories_2008-2012', 'Year_categories_2012-2017']


'''

Year_categories_2004-2008    0.459695
Year_categories_2008-2012    0.464162
Year_categories_2012-2016    0.491795
RoadCategory_TM              1.096948
RoadCategory_PR              1.557558
LinkLength_miles_scaled      4.228403

LinkLength_miles_scaled and RoadCategory_PR have the largest dependencies

''''

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#'''' USE K-NEAREST NEIGHBOUR TO FIND HOW ACCURATE OF THE MODEL IS TO FIT THE DATA

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# K-Nearest Neighbour

hyperparameters = {
    "n_neighbors": range(3,20,2),
    "weights": ["distance", "uniform"],
    "algorithm": ['brute'],
    "p": [1,2]
    }



# use K nearest neighbouR
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid=hyperparameters, cv=10)
grid.fit(df[columns_X],df[columns_y])
best_params = grid.best_params_
best_score = grid.best_score_

# The accuracy score for k-nearest neighbour is 0.90.
# Try and in=mprove this using Random Forrest

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#'''' USE RANDOM FORREST TO FIND HOW ACCURATE OF THE MODEL IS TO FIT THE DATA

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#use random forest

hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5, 10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [1, 5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
}

clf = RandomForestClassifier(random_state=1)
grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)
grid.fit(df[columns_X],df[columns_y])
best_params_rf = grid.best_params_
best_score_rf = grid.best_score_

# The accuracy score for Random Forrest is 0.907, which is better that K-Nearest Neighbor
# Therefore, we use Random Forrest to train the data and make predictions

holdout = create_holdout_data()

best_rf = grid.best_estimator_

input_holdout = holdout[['LinkLength_miles_scaled', 'RoadCategory_PR', 'RoadCategory_TM', 'Year_categories_2004-2008',
       'Year_categories_2008-2012', 'Year_categories_2012-2017']]

prediction = best_rf.predict(input_holdout)


compare_result = holdout[['Year','Road', 'RoadCategory', 'LinkLength_miles', 'CarsTaxis', 'CarsTaxis_categories']]
Prediction_Category =  pd.DataFrame({'Prediction_Category': [[prediction]] })
compare_result['Prediction_Category'] = prediction

conf_matrix = confusion_matrix(holdout['CarsTaxis_categories'], prediction)

