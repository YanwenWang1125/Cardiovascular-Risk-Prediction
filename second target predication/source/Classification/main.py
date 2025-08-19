import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
X_train = pd.read_parquet("./cardio_split/X_train.parquet")
Y_train = pd.read_parquet("./cardio_split/Y_train.parquet").squeeze()
X_validate = pd.read_parquet("./cardio_split/X_validate.parquet")
Y_validate = pd.read_parquet("./cardio_split/Y_validate.parquet").squeeze()
X_test = pd.read_parquet("./cardio_split/X_test.parquet")
Y_test = pd.read_parquet("./cardio_split/Y_test.parquet").squeeze()

# Combine training and validation sets for full training (exclude test set)
X_full_train = pd.concat([X_train, X_validate,X_test], axis=0, ignore_index=True)
Y_full_train = pd.concat([Y_train, Y_validate,Y_test], axis=0, ignore_index=True)

print(f"Full training data shape: {X_full_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Class distribution in full training set: {Y_full_train.value_counts()}")

# Advanced Feature Engineering
print("Performing advanced feature engineering...")

def create_interaction_features(df):
    df_new = df.copy()
    # Add domain-specific interactions for cardiovascular data
    if 'age' in df.columns and 'weight' in df.columns:
        df_new['age_weight_interaction'] = df['age'] * df['weight']
    if 'height' in df.columns and 'weight' in df.columns:
        df_new['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df_new['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df_new['mean_bp'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    return df_new

X_full_train_eng = create_interaction_features(X_full_train)
X_test_eng = create_interaction_features(X_test)
X_test_eng.to_parquet("./X_test.parquet")
'''
# Feature selection using mutual information
print("Selecting best features...")
selector = SelectKBest(score_func=mutual_info_classif, k=min(50, X_full_train_eng.shape[1]))
X_full_train_selected = selector.fit_transform(X_full_train_eng, Y_full_train)
X_test_selected = selector.transform(X_test_eng)

print(f"Selected {X_full_train_selected.shape[1]} features out of {X_full_train_eng.shape[1]}")

# Enhanced XGBoost with comprehensive hyperparameters
print("Setting up enhanced XGBoost model...")
enhanced_param_grid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [200, 400, 600, 800],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'colsample_bylevel': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.5, 1],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2, 5],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 2, 3]  # Handle class imbalance
}

xgb_enhanced = xgb.XGBClassifier(
    tree_method="hist",
    predictor="predictor",
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    verbosity=1,
    n_jobs=-1
)

# Use stratified k-fold for better CV
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search_enhanced = RandomizedSearchCV(
    xgb_enhanced,
    param_distributions=enhanced_param_grid,
    n_iter=100,  # More iterations for better search
    scoring="roc_auc",
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Starting hyperparameter tuning on full training set...")
search_enhanced.fit(X_full_train_selected, Y_full_train)
best_model = search_enhanced.best_estimator_

print(f"Best Parameters: {search_enhanced.best_params_}")
print(f"Best CV ROC-AUC Score: {search_enhanced.best_score_:.4f}")

# Save the best model
joblib.dump(best_model, "./best_xgb_model.pkl")
best_model.save_model("best_xgb_model.json")

print("\n" + "="*60)
print("FINAL EVALUATION - TEST SET ONLY")
print("="*60)

# Evaluate only on Test Set
test_preds = best_model.predict(X_test_selected)
test_probs = best_model.predict_proba(X_test_selected)[:, 1]

# Metrics for Test Set
roc_auc_test = roc_auc_score(Y_test, test_probs)
f1_test = f1_score(Y_test, test_preds)
accuracy_test = accuracy_score(Y_test, test_preds)

print(f"Test ROC-AUC: {roc_auc_test:.4f}")
print(f"Test F1 Score: {f1_test:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(Y_test, test_preds))

# Plot ROC Curve
print("Plotting ROC Curve...")
fpr, tpr, _ = roc_curve(Y_test, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f"XGBoost ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - XGBoost Model (Test Set)', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Final_ROC_Curve.png", dpi=300)
plt.show()

# Feature importance analysis
print("Creating feature importance plot with actual feature names...")

# Get the original feature names and add engineered features
original_features = ['height', 'weight', 'age_years', 'ap_hi', 'ap_lo', 'bmi', 'gender_2', 'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3', 'smoke_1', 'alco_1', 'active_1', 'bp_category_Hypertension_2', 'bp_category_Normal', 'bp_category_None']

# Add engineered feature names
engineered_features = []
if 'age_years' in original_features and 'weight' in original_features:
    engineered_features.append('age_weight_interaction')
if 'height' in original_features and 'weight' in original_features:
    engineered_features.append('bmi_engineered')  # Note: original dataset already has 'bmi'
if 'ap_hi' in original_features and 'ap_lo' in original_features:
    engineered_features.extend(['pulse_pressure', 'mean_bp'])

all_feature_names = original_features + engineered_features

# Get the selected feature indices and map them to actual names
selected_indices = selector.get_support(indices=True)
selected_feature_names = [all_feature_names[i] for i in selected_indices]

importances = best_model.feature_importances_

plt.figure(figsize=(14, 10))
indices = np.argsort(importances)[::-1][:20]  # Top 20 features

# Create the bar plot
bars = plt.bar(range(len(indices)), importances[indices], color='lightblue', edgecolor='navy', alpha=0.8)

# Customize the plot
plt.title('Top 20 Feature Importances - XGBoost Model', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=14, fontweight='bold')
plt.ylabel('Importance Score', fontsize=14, fontweight='bold')

# Set x-axis labels with actual feature names
feature_labels = [selected_feature_names[i] for i in indices]
plt.xticks(range(len(indices)), feature_labels, rotation=45, ha='right', fontsize=11)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Improve layout
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save as high-DPI vector image (SVG) and high-DPI PNG
plt.savefig("Feature_Importance.svg", format='svg', dpi=300, bbox_inches='tight')
plt.savefig("Feature_Importance.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Feature importance plot saved as:")
print(f"- Feature_Importance.svg (vector format)")
print(f"- Feature_Importance.png (300 DPI raster format)")

print(f"\nFinal Summary:")
print(f"Model: Enhanced XGBoost with Feature Engineering")
print(f"Training Data: Combined train + validation sets ({X_full_train_selected.shape[0]} samples)")
print(f"Test Set Size: {X_test_selected.shape[0]} samples")
print(f"Features Used: {X_full_train_selected.shape[1]} selected features")
print(f"Final Test ROC-AUC: {roc_auc_test:.4f}")
print(f"Final Test F1 Score: {f1_test:.4f}")
print(f"Final Test Accuracy: {accuracy_test:.4f}")
print("\nModel saved as: best_xgb_model.pkl and best_xgb_model.json")
'''