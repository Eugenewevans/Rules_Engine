import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error

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
X_train = df.loc[df['split'] == 'T', features]
X_test = df.loc[df['split'] == 'V', features]

y_train_binary = df.loc[df['split'] == 'T', binary_target]
y_test_binary = df.loc[df['split'] == 'V', binary_target]

y_train_reg = df.loc[df['split'] == 'T', numerical_target]
y_test_reg = df.loc[df['split'] == 'V', numerical_target]

# Define XGBoost models (no DMatrix)
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_child_weight=3,
    n_estimators=100,
    scale_pos_weight=1.0,
    random_state=42
)

xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_child_weight=3,
    n_estimators=100,
    random_state=42
)

# Train models
xgb_clf.fit(X_train, y_train_binary, eval_set=[(X_test, y_test_binary)], early_stopping_rounds=20, verbose=False)
xgb_reg.fit(X_train, y_train_reg, eval_set=[(X_test, y_test_reg)], early_stopping_rounds=20, verbose=False)

# ** Update DataFrame with predictions **
df.loc[df['split'] == 'V', 'class_pred'] = xgb_clf.predict_proba(X_test)[:, 1]  # Probability of class 1
df.loc[df['split'] == 'V', 'reg_pred'] = xgb_reg.predict(X_test)  # Regression prediction

# Compute Training & Validation Scores
train_logloss = log_loss(y_train_binary, xgb_clf.predict_proba(X_train)[:, 1])
valid_logloss = log_loss(y_test_binary, xgb_clf.predict_proba(X_test)[:, 1])

train_rmse = mean_squared_error(y_train_reg, xgb_reg.predict(X_train), squared=False)
valid_rmse = mean_squared_error(y_test_reg, xgb_reg.predict(X_test), squared=False)

# Print scores
print(f"🔵 Binary Classification:")
print(f" - Training Log Loss: {train_logloss:.4f}")
print(f" - Validation Log Loss: {valid_logloss:.4f}\n")

print(f"🟢 Regression Model:")
print(f" - Training RMSE: {train_rmse:.4f}")
print(f" - Validation RMSE: {valid_rmse:.4f}")

# Display updated DataFrame
import ace_tools as tools
tools.display_dataframe_to_user(name="Updated DataFrame with Predictions", dataframe=df)
