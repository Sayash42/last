 ### Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
 # Load the training dataset
train_df = pd.read_csv('train.csv')

# Load the test dataset
test_df = pd.read_csv('test.csv')
 # Label encode the entity values in the training dataset
le = LabelEncoder()
train_df['entity_value'] = le.fit_transform(train_df['entity_value'])

# Split the training dataset into features (X) and target (y)
X_train = train_df.drop('entity_value', axis=1)

# Iterate over all columns in X_train
for col in X_train.columns:
    # Check if the column contains string values
    if X_train[col].dtype == 'object':
        # Apply Label Encoding to the column
        X_train[col] = le.fit_transform(X_train[col])

y_train = train_df['entity_value']

# Split the test dataset into features (X)
X_test = test_df

# Iterate over all columns in X_test
for col in X_test.columns:
    # Check if the column contains string values
    if X_test[col].dtype == 'object':
        # Apply Label Encoding to the column
        X_test[col] = le.fit_transform(X_test[col])
 # Create and train the model (rest of your code)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 # Predict the entity values for the test dataset
y_pred = model.predict(X_test)

# Convert the predicted entity values back to their original labels
y_pred_labels = le.inverse_transform(y_pred)

# Create a new column in the test dataset with the predicted entity values
test_df['entity_value'] = y_pred_labels
 # Save the test dataset with the predicted entity values to a new CSV file
test_df.to_csv('test_with_entity.csv', index=False)




# Process CSV file in chunks
chunks = pd.read_csv('train.csv', chunksize=10000)
train_df = pd.concat(chunks)

# Apply Label Encoding for object (string) columns in one go
X_train = X_train.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)
X_test = X_test.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)
# Limit depth and/or reduce number of estimators for memory efficiency
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
import dask.dataframe as dd

# Use Dask to load and process large CSV files
train_df = dd.read_csv('train.csv')
test_df = dd.read_csv('test.csv')
import gc
gc.collect()
