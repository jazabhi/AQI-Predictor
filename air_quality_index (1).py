import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import pickle

from sklearn.metrics import mean_squared_error, r2_score


"""IMPORTING THE DATASET"""

df=pd.read_csv("C:/Users/KIIT/Documents/vs code/AQI prediction final/Air quality/air_quality1.csv")

print(df.head())

df.shape

df.tail()

df.describe()

df.info()

missing_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
print(f"The overall missing percentage in the DataFrame is: {missing_percentage:.2f}%")

"""**THEREFORE THERE ARE NULL VALUES IN MANY COLUMNS
 WE HAVE TO REMOVE IN DATA CLEANING**
"""

df["AQI_Bucket"].unique()

df["AQI_Bucket"].value_counts()

"""SINCE MODERATE VALUE in AQI_Bucket col is max so replace all the NAN values with the MODERATE"""

df['AQI_Bucket']=df['AQI_Bucket'].fillna('Moderate')

df.head()

numeric=df.select_dtypes(include=[float]).columns
numeric

df.isna().sum()

"""replacing all Nan values from numerical columns with median"""

for column in numeric:
  df[column]=df[column].fillna((df[column].median()))

cat_data=df.select_dtypes(include=[object])
cat_data.columns

df.head()

df.isna().sum()


"""Checking Co-Relation In Data"""

plt.figure(figsize=(10,8))
df['City'] = pd.to_numeric(df['City'], errors='coerce')  # Replace 'column_name' with the actual column name

"""CHECKING FOR DUPLICATES"""

df.duplicated().sum()

"""CONVERTING DATE ATTRITUBUTE TO DATE TIME DATATYPE"""

df['Date']=pd.to_datetime(df["Date"])
df['Date']


for column in numeric:
  # Calculate quartiles and IQR
  q1 = df[column].quantile(0.25)
  q3 = df[column].quantile(0.75)
  iqr = q3 - q1

  # Define the upper and lower bounds for outliers
  upper_bound = q3 + 3 * iqr
  lower_bound = q1 - 3 * iqr

  # Filter out the outliers
  outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
  if not outliers.empty:
    df.loc[outliers.index, column] = np.log(df.loc[outliers.index, column])

df.shape

print(df.head())


"""Categorical data encoding  ___  Feature Engineering"""
df['Date'] = pd.to_datetime(df['Date'])

encoded_city = pd.get_dummies(df['City'], prefix='City')
df = pd.concat([df, encoded_city], axis=1)

#from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['AQI_Bucket_Label'] = label_encoder.fit_transform(df['AQI_Bucket'])

df.dtypes

"""*Machine Learning Model Training:*"""

df. head(2)

# Extract the features (X) and target variable (y)
features = df.filter(['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2'],axis=1)
#features = df[columns = ['Year','Month','Day','PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2','AQI_Bucket_Label']]
target = df[['AQI']]
features.head(2)

"""Spliting The data into traning and testing dataset"""

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
pickle.dump(forest_model, open('forest_model.pkl', 'wb'))
forest_model=pickle.load(open('forest_model.pkl','rb'))