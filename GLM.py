import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# 1. Filter out rows where success_metric is null
df_filtered = df[df['success_metric'].notnull()].copy()

# 2. Create decile buckets
df_filtered['model1_bucket'] = pd.qcut(df_filtered['model1_output'], 10, labels=False)
df_filtered['model2_bucket'] = pd.qcut(df_filtered['model2_output'], 10, labels=False)

# 3. Prepare features and target
X = df_filtered[['model1_bucket', 'model2_bucket', 'your_categorical_feature']]
y = df_filtered['success_metric']  # binary target assumed

# 4. One-hot encode the categorical feature
categorical_features = ['your_categorical_feature']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # keep the model bucket features
)

# 5. Build the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
])

# 6. Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Fit the model
pipeline.fit(X_train, y_train)

# 8. Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Get feature names + coefficients
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist() + ['model1_bucket', 'model2_bucket']
coefficients = pipeline.named_steps['classifier'].coef_[0]

# 10. Output coefficients
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")
