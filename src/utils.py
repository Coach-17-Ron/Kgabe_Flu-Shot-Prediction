# CLASS DISTRIBUTION WITH PERCENTAGE

def plot_class_distribution(y, label):
    counts = y[label].value_counts().sort_index()
    percentages = counts / counts.sum() * 100
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette="pastel")
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        plt.text(i, count + 50, f'{count} ({pct:.1f}%)', ha='center')
    plt.title(f'Class Distribution for {label}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.ylim(0, counts.max() * 1.15)
    plt.show()

plot_class_distribution(y_full, 'h1n1_vaccine')
plot_class_distribution(y_full, 'seasonal_vaccine')


# FEATURE IMPORTANCE PLOT (Gain-based, Top 20)

# H1N1

# Extract feature names from OneHotEncoder after preprocessing
# This assumes you have access to your preprocessor to get feature names:
num_features = numeric_features
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_features = list(num_features) + list(cat_features)

# Get feature importance from LightGBM (for first target)
importance = final_model.estimators_[0].booster_.feature_importance(importance_type='gain')
feat_imp_df = pd.DataFrame({'feature': all_features, 'importance': importance})
feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=True).tail(20)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=feat_imp_df, palette="viridis")
plt.title('Top 20 Feature Importance (Gain) - H1N1 Vaccine')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# SEASONAL

# Extract feature names from OneHotEncoder after preprocessing
# This assumes you have access to your preprocessor to get feature names:
num_features = numeric_features
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_features = list(num_features) + list(cat_features)

# Get feature importance from LightGBM (for first target)
importance = final_model.estimators_[1].booster_.feature_importance(importance_type='gain')
feat_imp_df = pd.DataFrame({'feature': all_features, 'importance': importance})
feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=True).tail(20)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=feat_imp_df, palette="viridis")
plt.title('Top 20 Feature Importance (Gain) - Seasonal Vaccine')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# ROC CURVE FOR BOTH TARGETS

def plot_roc_curve(y_true, y_scores, label):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

plt.figure(figsize=(8,6))
plot_roc_curve(y_val['h1n1_vaccine'], lgbm_val_preds[0][:,1], 'H1N1 Vaccine')
plot_roc_curve(y_val['seasonal_vaccine'], lgbm_val_preds[1][:,1], 'Seasonal Vaccine')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Vaccine Predictions')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# CALIBRATION CURVE

def plot_calibration_curve(y_true, y_probs, label, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_probs[:, 1], n_bins=n_bins)
    plt.plot(prob_pred, prob_true, marker='o', label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(8,6))
plot_calibration_curve(y_val['h1n1_vaccine'], lgbm_val_preds[0], 'H1N1 Vaccine')
plot_calibration_curve(y_val['seasonal_vaccine'], lgbm_val_preds[1], 'Seasonal Vaccine')
plt.show()

