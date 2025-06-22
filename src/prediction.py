# FINAL MODEL TRAINING AND PREDICTION

# Preprocess all training data
X_full_processed = preprocessor.fit_transform(train_features)
y_full = train_labels[['h1n1_vaccine', 'seasonal_vaccine']]

# Train final LightGBM model on full training data
final_model = MultiOutputClassifier(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,            # allow deeper trees
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        num_leaves=40,
        min_split_gain=0.0,
        random_state=42,
        n_jobs=-1
    )
)

final_model.fit(X_full_processed, y_full)

# Preprocess test data
X_test_processed = preprocessor.transform(test_features)

# Make predictions on test set
test_preds = final_model.predict_proba(X_test_processed)

# Prepare submission file
submission_df = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'h1n1_vaccine': test_preds[0][:, 1],
    'seasonal_vaccine': test_preds[1][:, 1]
})

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")