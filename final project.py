#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'california_housing_data.csv'
housing_data = pd.read_csv(file_path)

# Display the first few rows and dataset information
housing_data_info = housing_data.info()
housing_data_head = housing_data.head()

housing_data_info, housing_data_head


# In[2]:


from sklearn.model_selection import train_test_split

# Define the features and target variable
X = housing_data.drop(columns=["MedHouseVal"])
y = housing_data["MedHouseVal"]

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes of the splits
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[3]:


import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from flaml import AutoML


# In[4]:


# Step 1: Define RMSE as the evaluation metric
def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Step 2: Set up MLflow experiment
mlflow.set_experiment("California Housing AutoML")

# Step 3: Train a model using FLAML's AutoML
automl = AutoML()
automl_settings = {
    "time_budget": 60,  # 1-minute budget for AutoML
    "metric": "rmse",
    "task": "regression",
    "log_file_name": "automl.log",
}

# Start an MLflow run
with mlflow.start_run():
    # Train the model
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    # Get the best model
    best_model = automl.model
    mlflow.sklearn.log_model(best_model, "best_model")
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculate RMSE
    rmse = calculate_rmse(y_test, y_pred)
    mlflow.log_metric("RMSE", rmse)

# Output RMSE and best model details
best_model_details = {
    "Best Algorithm": automl.best_estimator,
    "Best Configuration": automl.best_config,
    "Best RMSE": rmse,
}
best_model_details


# In[6]:


# Redefine and reload necessary components
from lightgbm import LGBMRegressor
import joblib
import os

# Reload the best model parameters (from earlier results)
best_model = LGBMRegressor(
    n_estimators=146,
    num_leaves=18,
    min_child_samples=3,
    learning_rate=0.17402065726724145,
    max_bin=2**8,
    colsample_bytree=0.6649148062238498,
    reg_alpha=0.0009765625,
    reg_lambda=0.006761362450996487,
)
# Fit the best model on the full training data
best_model.fit(X_train, y_train)

# Save the trained model
model_dir = "/mnt/data/deployed_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "lightgbm_best_model.pkl")
joblib.dump(best_model, model_path)

# Verify the model is saved
os.path.exists(model_path)


# In[7]:


# Load the saved model
loaded_model = joblib.load(model_path)

# Step 1: Predict on the original test dataset
y_pred_original = loaded_model.predict(X_test)

# Calculate RMSE for the original test dataset
rmse_original = calculate_rmse(y_test, y_pred_original)

# Step 2: Alter the test dataset (swap two features and randomize another)
X_test_altered = X_test.copy()
X_test_altered["AveRooms"], X_test_altered["AveBedrms"] = X_test_altered["AveBedrms"], X_test_altered["AveRooms"]
X_test_altered["Population"] = X_test_altered["Population"].sample(frac=1).values  # Random shuffle

# Predict on the altered test dataset
y_pred_altered = loaded_model.predict(X_test_altered)

# Calculate RMSE for the altered test dataset
rmse_altered = calculate_rmse(y_test, y_pred_altered)

# Return RMSE for both cases
rmse_results = {
    "RMSE (Original Test Data)": rmse_original,
    "RMSE (Altered Test Data)": rmse_altered,
}
rmse_results


# In[8]:


def make_prediction(input_data):
    """
    Simulate a deployed model's prediction.
    :param input_data: DataFrame or dictionary of features.
    :return: Model predictions.
    """
    prediction = loaded_model.predict(input_data)
    return prediction

# Example: Predict for a sample from the test dataset
sample_data = X_test.iloc[:1]
predicted_value = make_prediction(sample_data)
print(f"Predicted Value: {predicted_value}")


# In[9]:


import matplotlib.pyplot as plt

# Log predictions for original and altered data
original_predictions = loaded_model.predict(X_test)
altered_predictions = loaded_model.predict(X_test_altered)

# Compare predictions
plt.figure()
plt.scatter(y_test, original_predictions, label="Original Data Predictions", alpha=0.6)
plt.scatter(y_test, altered_predictions, label="Altered Data Predictions", alpha=0.6, color='red')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Prediction Comparison")
plt.show()


# In[10]:


# Simulate data drift by altering feature distributions
X_test_drifted = X_test.copy()
X_test_drifted["MedInc"] *= 1.2  # Increase median income by 20%

# Predict on drifted data
drifted_predictions = loaded_model.predict(X_test_drifted)

# Compare drifted predictions with original
plt.figure()
plt.hist(original_predictions, alpha=0.5, label="Original")
plt.hist(drifted_predictions, alpha=0.5, label="Drifted", color="orange")
plt.legend()
plt.title("Prediction Distribution: Original vs Drifted")
plt.show()

