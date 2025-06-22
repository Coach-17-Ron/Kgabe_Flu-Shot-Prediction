# FIRST MODEL = RANDOM FOREST

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_features, 
    train_labels[['h1n1_vaccine', 'seasonal_vaccine']],
    test_size=0.2,
    random_state=42
)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

# Evaluate on validation set
rf_val_preds = rf_model.predict_proba(X_val_processed)
rf_h1n1_auc = roc_auc_score(y_val['h1n1_vaccine'], rf_val_preds[0][:, 1])
rf_seasonal_auc = roc_auc_score(y_val['seasonal_vaccine'], rf_val_preds[1][:, 1])
rf_mean_auc = (rf_h1n1_auc + rf_seasonal_auc) / 2

print(f"Random Forest Validation AUC - H1N1: {rf_h1n1_auc:.4f}")
print(f"Random Forest Validation AUC - Seasonal: {rf_seasonal_auc:.4f}")
print(f"Random Forest Mean Validation AUC: {rf_mean_auc:.4f}")

# SECOND MODEL = XGBOOST
# XGBoost model with MultiOutputClassifier
xgb_model = MultiOutputClassifier(
    XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
)

xgb_model.fit(X_train_processed, y_train)

# Evaluate on validation set
xgb_val_preds = xgb_model.predict_proba(X_val_processed)
xgb_h1n1_auc = roc_auc_score(y_val['h1n1_vaccine'], xgb_val_preds[0][:, 1])
xgb_seasonal_auc = roc_auc_score(y_val['seasonal_vaccine'], xgb_val_preds[1][:, 1])
xgb_mean_auc = (xgb_h1n1_auc + xgb_seasonal_auc) / 2

print(f"XGBoost Validation AUC - H1N1: {xgb_h1n1_auc:.4f}")
print(f"XGBoost Validation AUC - Seasonal: {xgb_seasonal_auc:.4f}")
print(f"XGBoost Mean Validation AUC: {xgb_mean_auc:.4f}")


# THIRD MODEL = LOGISTIC REGRESSION
# Logistic Regression model
lr_model = MultiOutputClassifier(
    LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
)

lr_model.fit(X_train_processed, y_train)

# Evaluate on validation set
lr_val_preds = lr_model.predict_proba(X_val_processed)
lr_h1n1_auc = roc_auc_score(y_val['h1n1_vaccine'], lr_val_preds[0][:, 1])
lr_seasonal_auc = roc_auc_score(y_val['seasonal_vaccine'], lr_val_preds[1][:, 1])
lr_mean_auc = (lr_h1n1_auc + lr_seasonal_auc) / 2

print(f"Logistic Regression Validation AUC - H1N1: {lr_h1n1_auc:.4f}")
print(f"Logistic Regression Validation AUC - Seasonal: {lr_seasonal_auc:.4f}")
print(f"Logistic Regression Mean Validation AUC: {lr_mean_auc:.4f}")

# FOUTRTH MODEL = LIGHTGBM
# LightGBM model with MultiOutputClassifier
lgbm_model = MultiOutputClassifier(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        num_leaves=40,
        min_split_gain=0.0,
        random_state=42,
        n_jobs=-1
    )
)

# Fit the model
lgbm_model.fit(X_train_processed, y_train)

# Predict probabilities
lgbm_val_preds = lgbm_model.predict_proba(X_val_processed)

# Evaluate AUC
lgbm_h1n1_auc = roc_auc_score(y_val['h1n1_vaccine'], lgbm_val_preds[0][:, 1])
lgbm_seasonal_auc = roc_auc_score(y_val['seasonal_vaccine'], lgbm_val_preds[1][:, 1])
lgbm_mean_auc = (lgbm_h1n1_auc + lgbm_seasonal_auc) / 2

# Print results
print(f"LightGBM Validation AUC - H1N1: {lgbm_h1n1_auc:.4f}")
print(f"LightGBM Validation AUC - Seasonal: {lgbm_seasonal_auc:.4f}")
print(f"LightGBM Mean Validation AUC: {lgbm_mean_auc:.4f}")