import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Sample DataFrame
df = pd.DataFrame({
    'feature1': range(100),
    'feature2': range(100, 200),
    'binary_target': [1 if i % 2 == 0 else 0 for i in range(100)],
    'numerical_target': [i * 1.5 for i in range(100)]
})

# Define feature lists
features = ['feature1', 'feature2']
binary_target = 'binary_target'
numerical_target = 'numerical_target'

# Split the dataset and assign 'T' or 'V' labels
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
df['split'] = 'V'  # Default to validation
df.loc[train_idx, 'split'] = 'T'  # Mark training set

# Define train and test sets based on the split column
X_train = df[features][df['split'] == 'T']
X_test = df[features][df['split'] == 'V']

y_train_binary = df[binary_target][df['split'] == 'T']
y_test_binary = df[binary_target][df['split'] == 'V']

y_train_reg = df[numerical_target][df['split'] == 'T']
y_test_reg = df[numerical_target][df['split'] == 'V']

# Convert datasets into DMatrix format (preferred for early stopping)
dtrain_binary = xgb.DMatrix(X_train, label=y_train_binary)
dvalid_binary = xgb.DMatrix(X_test, label=y_test_binary)

dtrain_reg = xgb.DMatrix(X_train, label=y_train_reg)
dvalid_reg = xgb.DMatrix(X_test, label=y_test_reg)

# Define common XGBoost parameters
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'min_child_weight': 3,
    'n_estimators': 100,
    'random_state': 42
}

# ** Binary Classification Model **
xgb_clf_params = xgb_params.copy()
xgb_clf_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'scale_pos_weight': 1.0  # Adjust for imbalanced datasets if needed
})

xgb_clf = xgb.train(
    params=xgb_clf_params,
    dtrain=dtrain_binary,
    num_boost_round=1000,  # Maximum number of boosting rounds
    evals=[(dtrain_binary, 'train'), (dvalid_binary, 'eval')],
    early_stopping_rounds=20,  # Stops if no improvement after 20 rounds
    verbose_eval=50
)

# Predict and evaluate binary classification model
y_pred_binary = (xgb_clf.predict(dvalid_binary) > 0.5).astype(int)
binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"Binary Classification Accuracy: {binary_accuracy:.4f}")

# ** Regression Model **
xgb_reg_params = xgb_params.copy()
xgb_reg_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
})

xgb_reg = xgb.train(
    params=xgb_reg_params,
    dtrain=dtrain_reg,
    num_boost_round=1000,  # Maximum number of boosting rounds
    evals=[(dtrain_reg, 'train'), (dvalid_reg, 'eval')],
    early_stopping_rounds=20,  # Stops if no improvement after 20 rounds
    verbose_eval=50
)

# Predict and evaluate regression model
y_pred_reg = xgb_reg.predict(dvalid_reg)
regression_mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Regression Mean Squared Error: {regression_mse:.4f}")
