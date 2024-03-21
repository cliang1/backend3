import seaborn as sns
# Load the titanic dataset
exercise_data = sns.load_dataset('exercise')
print("Exercise Data")
print(exercise_data.columns) # titanic data set
display(exercise_data[['id', 'diet' ,'pulse' ,'time','kind']])

import seaborn as sns
import pandas as pd
# Preprocess the data
from sklearn.preprocessing import OneHotEncoder

# Preprocessing steps
td = exercise_data.copy()  # Make a copy to avoid modifying the original data
td.dropna(inplace=True) # drop rows with at least one missing value
td['id'] = td['id'].astype('category').cat.codes # Encoding id as categorical
td['diet'] = td['diet'].astype('category').cat.codes # Encoding diet as categorical
td['time'] = td['time'].astype('category').cat.codes # Encoding time as categorical
td['kind'] = td['kind'].astype('category').cat.codes # Encoding kind as categorical

print(td.columns)
display(td)

# Median
# Calculate the median of the category codes for the 'kind' column
print("id:")
median_id_cat_code= td['id'].median()
print(median_id_cat_code)
print("diet:")
median_diet_cat_code= td['diet'].median()
print(median_diet_cat_code)
print("pulse:")
median_pulse_cat_code= td['pulse'].median()
print(median_pulse_cat_code)
print("time:")
median_time_cat_code= td['time'].median()
print(median_time_cat_code)
print("kind:")
median_kind_cat_code= td['kind'].median()
print(median_kind_cat_code)

# Mean
print("id:")
mean_id_cat_code= td['id'].mean()
print(mean_id_cat_code)
print("diet:")
mean_diet_cat_code= td['diet'].mean()
print(mean_diet_cat_code)
print("pulse:")
mean_pulse_cat_code= td['pulse'].mean()
print(mean_pulse_cat_code)
print("time:")
mean_time_cat_code= td['time'].mean()
print(mean_time_cat_code)
print("kind:")
mean_kind_cat_code= td['kind'].mean()
print(mean_kind_cat_code)

# Select numerical columns
numeric_columns = exercise_data.select_dtypes(include=['number'])
print("Maximums score")
print(numeric_columns[exercise_data['pulse'] == 110].max())
print()
print("Minimums score")
print(numeric_columns[exercise_data['pulse'] == 83].min())

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the exercise dataset using Seaborn
exercise_data = sns.load_dataset('exercise')

# Display the columns and a sample of the dataset
print("Exercise Data Columns:")
print(exercise_data.columns)
print("\nSample of the Exercise Data:")
print(exercise_data[['id', 'diet', 'pulse', 'time', 'kind']].head())

# Split the data into features (X) and target (y)
X = exercise_data[[ 'id', 'diet', 'time', 'kind']]
y = exercise_data['pulse']

# Perform one-hot encoding for the categorical columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['diet', 'time', 'kind'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Print the model coefficients
print("Model Coefficients:")
print(regressor.coef_)

# Print the model intercept
print("Model Intercept:")
print(regressor.intercept_)

# Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Load the exercise dataset
exercise_data = pd.read_csv("../exercise.csv")
# Define a new passenger
passenger = pd.DataFrame({
    'id': [1],  # Choose an ID that does not exist in the original dataset
    'diet': ['low fat'],
    'time': ['1 min'],
    'kind': ['rest']
})
# Preprocess the new passenger data
new_passenger = pd.get_dummies(passenger, columns=['diet', 'time', 'kind'])
# Combine the new passenger's data with the original dataset
combined_data = pd.concat([exercise_data, new_passenger], ignore_index=True)
# Split the data into features (X) and target (y)
X = combined_data[['id', 'diet', 'time', 'kind']]
y = combined_data['pulse']
# Perform one-hot encoding for the categorical columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['diet', 'time', 'kind'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)
# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_encoded[:-1], y[:-1])  # Exclude the last row which is the new passenger's data
# Predict the pulse rate for the new passenger
pulse_prediction = regressor.predict(X_encoded[-1:])  # Predict for the last row which is the new passenger's data
# Round off the predicted pulse rate to two decimal places
rounded_pulse_prediction = round(pulse_prediction[0], 2)
# Print the predicted pulse rate
print('Predicted Pulse Rate:', rounded_pulse_prediction)

