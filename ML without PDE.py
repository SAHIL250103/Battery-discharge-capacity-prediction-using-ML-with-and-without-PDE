# %% [markdown]
# #Included Libraries

# %%
# This Code is utilized to make an Optimized Machine Learning Model by Training, Testing, and Validating the Battery dataset.
# The ML models used here include SVR, DT, RF,  GBR, and GPR. And for each model a performance matrix was determined as a measuring quantity to determine the models accuracy and errors while making predictions
# These performance matrix include MSE, RMSE, MAE, and R2 score, which are then represented using bar graph.
# Also the actual and predicted values of each model has been obtained and visualized on a scatterplot for each model for their training set, testing and validation set as well.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.model_selection import cross_validate

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

# %% [markdown]
# ##CSV File Load

# %%
# Read the dataset
df = pd.read_csv('/content/Dataset.csv')
df.head()

# %% [markdown]
# #Data Visualization

# %% [markdown]
# ##HeatMaps: (Pearson And Spearman)

# %%
# Calculate Pearson correlation
corr_pearson = df.corr(method='pearson')

# Calculate Spearman correlation
corr_spearman = df.corr(method='spearman')

# Plot Pearson correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation')
plt.savefig('pearson_heatmap.pdf', format='pdf')  # Save the heatmap as a PDF file
plt.show()

# Plot Spearman correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation')
plt.savefig('spearman_heatmap.pdf', format='pdf')  # Save the heatmap as a PDF file
plt.show()

# %% [markdown]
# ##Scatter Plot

# %%
# Create the scatter plot
scatter = plt.scatter(df['cycle number'], df['Q discharge/mA.h'], c=df['C-rate'])

# Add a legend
unique_c_rates = df['C-rate'].unique()
handles, labels = scatter.legend_elements()
plt.legend(handles, unique_c_rates, title='C-rate')

# Set labels for the axes
plt.xlabel('Cycle Number')
plt.ylabel('Q discharge/mA.h')

# Show the plot
plt.show()

# %%
dfn = pd.read_csv('/content/Combined Dataset.csv', usecols=['cycle number', 'time/s', 'Ecell/V', '<I>/mA', 'Temperature', 'C-rate', 'Q discharge/mA.h'])

x = dfn.iloc[:, 0:6]
y = dfn.iloc[:, -1]

# %% [markdown]
# #Train-Test-Split-Validation

# %%
# Divide the dataset into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=30)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=30)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# %% [markdown]
# #Models

# %% [markdown]
# ##Decision Tree Regressor

# %%
# Define the parameter grid
param_grid = {
    "splitter": ["best", "random"],
    "max_depth": [1, 5, 9, 12],
    "min_samples_leaf": [3, 5, 7, 9],
    "min_weight_fraction_leaf": [0.2, 0.4],
    "max_features": ["auto", "log2", "sqrt", None],
    "max_leaf_nodes": [20, 40, 80]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

# %%
# Calculate predictions on train, test, and validation sets
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)
y_pred_val = best_model.predict(X_val_scaled)

# %%
# Calculate evaluation metrics on the train set
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print("Train set - MSE:", mse_train, "MAE:", mae_train, "RMSE:", rmse_train, "R2:", r2_train)

# %%
# Calculate evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("Test set - MSE:", mse_test, "MAE:", mae_test, "RMSE:", rmse_test, "R2:", r2_test)

# %%
# Calculate evaluation metrics on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)

print("Validation set - MSE:", mse_val, "MAE:", mae_val, "RMSE:", rmse_val, "R2:", r2_val)

# %%
DTR_CV = cross_validate(DecisionTreeRegressor(),X_train_scaled,y_train,cv=5,scoring=('r2', 'neg_mean_squared_error'))
print(DTR_CV)

# %%
# Generate row numbers
row_numbers_train = range(1, len(y_pred_train) + 1)
row_numbers_test = range(1, len(y_pred_test) + 1)
row_numbers_val = range(1, len(y_pred_val) + 1)

# Scatter plot for train set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_train, y_train, color='blue', label='Actual')
plt.scatter(row_numbers_train, y_pred_train, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Train Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for test set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_test, y_test, color='green', label='Actual')
plt.scatter(row_numbers_test, y_pred_test, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for validation set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_val, y_val, color='red', label='Actual')
plt.scatter(row_numbers_val, y_pred_val, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Validation Set - Actual vs Predicted')
plt.legend()
plt.show()

# %% [markdown]
# ##Random Forest

# %%
# Random Forest Regressor

# Define the parameter grid
param_grid = {
    'bootstrap': [True],
    'max_depth': [2, 3],
    'max_features': [1, 2],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [4, 6, 8],
    'n_estimators': [100, 200, 300]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

# %%
# Calculate predictions on train, test, and validation sets
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)
y_pred_val = best_model.predict(X_val_scaled)

# %%
# Calculate evaluation metrics on the train set
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print("Train set - MSE:", mse_train, "MAE:", mae_train, "RMSE:", rmse_train, "R2:", r2_train)

# %%
# Calculate evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("Test set - MSE:", mse_test, "MAE:", mae_test, "RMSE:", rmse_test, "R2:", r2_test)

# %%
# Calculate evaluation metrics on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)

print("Validation set - MSE:", mse_val, "MAE:", mae_val, "RMSE:", rmse_val, "R2:", r2_val)

# %%
RF_CV = cross_validate(RandomForestRegressor(),X_train_scaled,y_train,cv=5,scoring=('r2', 'neg_mean_squared_error'))
print(RF_CV)

# %%
# Generate row numbers
row_numbers_train = range(1, len(y_pred_train) + 1)
row_numbers_test = range(1, len(y_pred_test) + 1)
row_numbers_val = range(1, len(y_pred_val) + 1)

# Scatter plot for train set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_train, y_train, color='blue', label='Actual')
plt.scatter(row_numbers_train, y_pred_train, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Train Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for test set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_test, y_test, color='green', label='Actual')
plt.scatter(row_numbers_test, y_pred_test, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for validation set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_val, y_val, color='red', label='Actual')
plt.scatter(row_numbers_val, y_pred_val, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Validation Set - Actual vs Predicted')
plt.legend()
plt.show()

# %% [markdown]
# ##Support Vector Regressor

# %%
# Define the parameter grid
param_grid = {
    'kernel': ('poly', 'rbf', 'sigmoid'),
    'C': [1, 5, 10],
    'degree': [3, 8],
    'coef0': [0.1, 10, 0.5],
    'gamma': ('auto', 'scale')
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

# %%
# Calculate predictions on train, test, and validation sets
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)
y_pred_val = best_model.predict(X_val_scaled)

# %%
# Calculate evaluation metrics on the train set
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print("Train set - MSE:", mse_train, "MAE:", mae_train, "RMSE:", rmse_train, "R2:", r2_train)

# %%
# Calculate evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("Test set - MSE:", mse_test, "MAE:", mae_test, "RMSE:", rmse_test, "R2:", r2_test)

# %%
# Calculate evaluation metrics on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)

print("Validation set - MSE:", mse_val, "MAE:", mae_val, "RMSE:", rmse_val, "R2:", r2_val)

# %%
SVR_CV = cross_validate(SVR(),X_train_scaled,y_train,cv=5,scoring=('r2', 'neg_mean_squared_error'))
print(SVR_CV)

# %%
# Generate row numbers
row_numbers_train = range(1, len(y_pred_train) + 1)
row_numbers_test = range(1, len(y_pred_test) + 1)
row_numbers_val = range(1, len(y_pred_val) + 1)

# Scatter plot for train set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_train, y_train, color='blue', label='Actual')
plt.scatter(row_numbers_train, y_pred_train, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Train Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for test set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_test, y_test, color='green', label='Actual')
plt.scatter(row_numbers_test, y_pred_test, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for validation set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_val, y_val, color='red', label='Actual')
plt.scatter(row_numbers_val, y_pred_val, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Validation Set - Actual vs Predicted')
plt.legend()
plt.show()

# %% [markdown]
# ##Gaussian Process Regressor

# %%
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor

# Define the parameter grid
param_grid = {
    "alpha":  [1, 10],
    "kernel": [RBF(l) for l in np.logspace(1, 2)],
}

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=GaussianProcessRegressor(),
    param_grid=param_grid,
    cv=2,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

# %%
# Calculate predictions on train, test, and validation sets
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)
y_pred_val = best_model.predict(X_val_scaled)

# %%
# Calculate evaluation metrics on the train set
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print("Train set - MSE:", mse_train, "MAE:", mae_train, "RMSE:", rmse_train, "R2:", r2_train)

# %%
# Calculate evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("Test set - MSE:", mse_test, "MAE:", mae_test, "RMSE:", rmse_test, "R2:", r2_test)

# %%
# Calculate evaluation metrics on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)

print("Validation set - MSE:", mse_val, "MAE:", mae_val, "RMSE:", rmse_val, "R2:", r2_val)

# %%
GPR_CV = cross_validate(GaussianProcessRegressor(),X_train_scaled,y_train,cv=5,scoring=('r2', 'neg_mean_squared_error'))
print(GPR_CV)

# %%
# Generate row numbers
row_numbers_train = range(1, len(y_pred_train) + 1)
row_numbers_test = range(1, len(y_pred_test) + 1)
row_numbers_val = range(1, len(y_pred_val) + 1)

# Scatter plot for train set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_train, y_train, color='blue', label='Actual')
plt.scatter(row_numbers_train, y_pred_train, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Train Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for test set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_test, y_test, color='green', label='Actual')
plt.scatter(row_numbers_test, y_pred_test, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for validation set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_val, y_val, color='red', label='Actual')
plt.scatter(row_numbers_val, y_pred_val, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Validation Set - Actual vs Predicted')
plt.legend()
plt.show()

# %% [markdown]
# ##Gradient Boosting Regressor

# %%
param_grid = {
    'learning_rate': [1.97, 1.98],
    'subsample': [0.9, 1.0],
    'max_depth': [50, 100, 150]
}

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(),
    param_grid=param_grid,
    cv=2,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

# %%
# Calculate predictions on train, test, and validation sets
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)
y_pred_val = best_model.predict(X_val_scaled)

# %%
# Calculate evaluation metrics on the train set
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print("Train set - MSE:", mse_train, "MAE:", mae_train, "RMSE:", rmse_train, "R2:", r2_train)

# %%
# Calculate evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("Test set - MSE:", mse_test, "MAE:", mae_test, "RMSE:", rmse_test, "R2:", r2_test)

# %%
# Calculate evaluation metrics on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)

print("Validation set - MSE:", mse_val, "MAE:", mae_val, "RMSE:", rmse_val, "R2:", r2_val)

# %%
GBR_CV = cross_validate(GradientBoostingRegressor(),X_train_scaled,y_train,cv=5,scoring=('r2', 'neg_mean_squared_error'))
print(GBR_CV)

# %%
# Generate row numbers
row_numbers_train = range(1, len(y_pred_train) + 1)
row_numbers_test = range(1, len(y_pred_test) + 1)
row_numbers_val = range(1, len(y_pred_val) + 1)

# Scatter plot for train set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_train, y_train, color='blue', label='Actual')
plt.scatter(row_numbers_train, y_pred_train, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Train Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for test set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_test, y_test, color='green', label='Actual')
plt.scatter(row_numbers_test, y_pred_test, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Test Set - Actual vs Predicted')
plt.legend()
plt.show()

# Scatter plot for validation set
plt.figure(figsize=(8, 6))
plt.scatter(row_numbers_val, y_val, color='red', label='Actual')
plt.scatter(row_numbers_val, y_pred_val, color='orange', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target Value')
plt.title('Validation Set - Actual vs Predicted')
plt.legend()
plt.show()


