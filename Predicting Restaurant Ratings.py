#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[25]:


df=pd.read_csv('Dataset.csv')


# In[26]:


df.head()


# In[27]:


df.dropna(inplace=True)


# # Converting categorical data into numerical formats that the model can understand.(One Hot Encoding)

# In[28]:


# Label Encoding for binary categorical features
label_encoder = LabelEncoder()
df['Has Table booking'] = label_encoder.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = label_encoder.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = label_encoder.fit_transform(df['Is delivering now'])
df['Switch to order menu'] = label_encoder.fit_transform(df['Switch to order menu'])

# One-Hot Encoding for multi-class categorical features
df = pd.get_dummies(df, columns=['Currency', 'City', 'Cuisines', 'Rating color', 'Rating text'])


# ### Now we  selected specific numerical and binary features along with one-hot encoded categorical features from the DataFrame to create the feature matrix (`features`), and extracted the target variable (`Aggregate rating`) to form the target vector (`target`).

# In[29]:


features = df[['Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 'Has Table booking',
               'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Price range', 
               'Votes'] + list(df.columns[df.columns.str.startswith('Currency_')]) +
              list(df.columns[df.columns.str.startswith('City_')]) + 
              list(df.columns[df.columns.str.startswith('Cuisines_')]) + 
              list(df.columns[df.columns.str.startswith('Rating color_')]) + 
              list(df.columns[df.columns.str.startswith('Rating text_')])]

target = df['Aggregate rating']


# ### We split the data into training and testing sets, with 80% of the data used for training and 20% for testing, to evaluate the model's performance on unseen data.

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ### We standardized the feature data by fitting the `StandardScaler` on the training set and then transforming both the training and testing sets to ensure they have a mean of 0 and a standard deviation of 1.

# In[31]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Training a Random Forest Regressor with 100 trees on the standardized training data.

# ##### We used Random Forest because it handles complex, non-linear relationships well, provides feature importance insights, and is robust to overfitting and noisy data, making it suitable for predicting the aggregate rating in this case. 

# In[32]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# ## Using the trained Random Forest model to make predictions on the standardized test data.

# In[33]:


y_pred = model.predict(X_test_scaled)


# #### Now the Mean Squared Error and R² Score to evaluate the model's prediction accuracy and explained variance, showing a low error and high explanatory power.

# In[34]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# ##### The output indicates that the model has a low Mean Squared Error (0.0308), meaning its predictions are close to the actual values, and a high R² Score (0.987), indicating that the model explains approximately 98.7% of the variance in the target variable, reflecting strong predictive performance.

# ### Now preparing new data by adding any missing columns and ensuring column order matches the training set, then scaled it and used the trained model to make and print predictions.

# In[35]:


new_data = df

# Add missing one-hot encoded columns with zeros
for col in X_train.columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Ensure the order of columns matches
new_data = new_data[X_train.columns]

# Transform the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions using the trained model
predictions = model.predict(new_data_scaled)

# Print the predictions
print(predictions)


# ### Now performing 5-fold cross-validation to evaluate the model's performance on different subsets of the training data, and printed the individual R² scores and their average to assess the model's consistency and reliability.

# In[36]:


from sklearn.model_selection import cross_val_score
# Perform 5-fold cross-validation
cv_scores = cross_val_score( model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-validation R² scores: {cv_scores}')
print(f'Average cross-validation R² score: {cv_scores.mean()}')


# In[37]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# Best parameters from the GridSearch
print(f'Best parameters: {grid_search.best_params_}')


# In[38]:


best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Evaluate the tuned model
y_pred_tuned = best_model.predict(X_test_scaled)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f'Tuned Mean Squared Error: {mse_tuned}')
print(f'Tuned R² Score: {r2_tuned}')


# In[40]:


import matplotlib.pyplot as plt

# Get feature importances
feature_importances = best_model.feature_importances_
feature_names = features.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Restaurant Rating Prediction')
plt.gca().invert_yaxis()
plt.show()


# In[41]:


import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))  # Increased size for better readability
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Restaurant Rating Prediction')
plt.gca().invert_yaxis()  # Highest importance at the top

# Adding value labels on the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}',
             va='center', ha='left', fontsize=10, color='black')

plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.tight_layout()  # Adjust layout to avoid clipping of labels
plt.show()


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define your model
model = LinearRegression()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-validation R² scores: {cv_scores}')
print(f'Average cross-validation R² score: {cv_scores.mean()}')

# Visualize the cross-validation R² scores
plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores, vert=False)
plt.title('Cross-validation R² scores')
plt.xlabel('R² score')plt.ylabel('Cross-validation')
plt.show()


# Description of the Visualization
# The boxplot generated from the code above shows the distribution of R² scores obtained from 5-fold cross-validation. Here's what the visualization typically tells us:
# 
# Central Tendency: The line inside the box represents the median R² score. This gives an idea of the typical performance of the model across the folds.
# Spread: The edges of the box represent the first and third quartiles (Q1 and Q3), giving an idea of the spread of the central 50% of the R² scores.
# Whiskers: The whiskers extend from the box to the smallest and largest values within 1.5 times the interquartile range (IQR) from Q1 and Q3, respectively. This indicates the range of most of the R² scores.
# Outliers: Any points outside the whiskers are considered outliers. These indicate folds where the model's performance was significantly different from the rest.
# Overall, the boxplot helps you quickly assess the variability and central tendency of your model's performance across different folds in cross-validation.

# In[ ]:




