# -------------------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules for preprocessing, feature selection, and modeling
from sklearn.preprocessing import LabelEncoder  
from sklearn.feature_selection import SelectKBest, mutual_info_classif  
from sklearn.model_selection import train_test_split  


sns.set_theme(style="whitegrid")  
plt.rcParams['figure.figsize'] = (12, 6)  
%matplotlib inline  

print(" All libraries imported successfully!")

# -------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------

df_raw = sns.load_dataset('titanic')

df = df_raw.copy()

print("First 5 rows of the dataset:")
display(df.head())
print(f"\nDataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")

# -------------------------------------------
# 3. EXPLORE DATASET (EDA)
# -------------------------------------------

# 3.1 Data Types & Non-Null Counts

print("="*50)
print("DATA TYPES & MISSING VALUES OVERVIEW")
print("="*50)
print(df.info())
print("\n")  # Add space

# 3.2 Statistical Summary for Numerical Columns

print("="*50)
print("STATISTICAL SUMMARY (NUMERICAL FEATURES)")
print("="*50)
display(df.describe())

# 3.3 Statistical Summary for Categorical Columns

print("="*50)
print("STATISTICAL SUMMARY (CATEGORICAL FEATURES)")
print("="*50)
display(df.describe(include=['object', 'category']))

# 3.4 Check for Missing Values Explicitly

print("="*50)
print("MISSING VALUES PER COLUMN")
print("="*50)
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing %': missing_percentage
}).sort_values(by='Missing %', ascending=False)
display(missing_df[missing_df['Missing Count'] > 0])  

# 3.5 Visualize Missing Data using a Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Heatmap of Missing Values (Yellow = Missing)', fontsize=14)
plt.show()

# 3.6 Target Variable Distribution (Survival Rate)

plt.figure(figsize=(8, 5))
sns.countplot(x='survived', data=df, palette='Set2')
plt.title('Distribution of Target Variable (0 = Died, 1 = Survived)', fontsize=14)
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
print(f"Overall Survival Rate: {df['survived'].mean() * 100:.2f}%")

# 3.7 Explore Relationship between Categorical Features and Target

# Pclass vs Survival
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.barplot(x='pclass', y='survived', data=df, palette='pastel')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')

# Sex vs Survival
plt.subplot(1, 3, 2)
sns.barplot(x='sex', y='survived', data=df, palette='pastel')
plt.title('Survival Rate by Gender')

# Embarked vs Survival
plt.subplot(1, 3, 3)
sns.barplot(x='embarked', y='survived', data=df, palette='pastel')
plt.title('Survival Rate by Port of Embarkation')
plt.tight_layout()
plt.show()

# 3.8 Explore Relationship between Numerical Features and Target
# Age distribution for survivors vs non-survivors
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='age', hue='survived', kde=True, bins=20, palette='Set1')
plt.title('Age Distribution by Survival Status')

# Fare distribution for survivors vs non-survivors
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='fare', hue='survived', kde=True, bins=20, palette='Set1')
plt.title('Fare Distribution by Survival Status')
plt.tight_layout()
plt.show()

# -------------------------------------------
# 4. CLEAN DATA
# -------------------------------------------

# 4.1 Drop Highly Redundant or Unusable Columns
columns_to_drop = ['deck', 'embark_town', 'alive', 'who', 'adult_male', 'alone', 'name', 'ticket']
df.drop(columns=columns_to_drop, inplace=True, axis=1)
print(f"Columns after dropping redundant features: {df.columns.tolist()}")

# 4.2 Handle Missing Values

age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)
print(f"Imputed missing 'age' values with median: {age_median}")

embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)
print(f"Imputed missing 'embarked' values with mode: {embarked_mode}")

fare_median = df['fare'].median()
df['fare'] = df['fare'].fillna(fare_median)
print(f"Imputed missing 'fare' with median: {fare_median}")

# 4.3 Verify that all missing values are handled

print("\nRemaining missing values after cleaning:")
print(df.isnull().sum().sum())  # Should be 0

# 4.4 Feature Engineering (Creating new helpful features)

# Create 'Family_Size' = siblings/spouse + parents/children + 1 (self)
df['Family_Size'] = df['sibsp'] + df['parch'] + 1

# Create 'Is_Alone' (binary) - though we can use 1 if Family_Size == 1 else 0
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)

print("\nNew features created: 'Family_Size' and 'Is_Alone'")
display(df[['sibsp', 'parch', 'Family_Size', 'Is_Alone']].head())

# -------------------------------------------
# 5. FEATURE SELECTION
# -------------------------------------------

# 5.1 Encode Categorical Variables to Numeric for Correlation
# -------------------------------------------
# 'sex' -> 0 for male, 1 for female
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})

# 'embarked' -> One-Hot Encoding is safer, but Label Encoding works for ordinal/ranking
# Since ports have no ordinal relation, we will use One-Hot Encoding via pandas get_dummies
df = pd.get_dummies(df, columns=['embarked'], prefix='port', drop_first=True)  # drop_first avoids multicollinearity

# Drop the original raw text columns (we have encoded versions)
df.drop(columns=['sex'], inplace=True)  # 'sex' is replaced by 'sex_encoded'

print("Categorical features successfully encoded to numeric.")
display(df.head())

# 5.2 Correlation Matrix (Filter Method)
# -------------------------------------------
# Compute correlation of all numerical features with the target 'survived'
correlation_matrix = df.corr(numeric_only=True)
target_correlation = correlation_matrix['survived'].sort_values(ascending=False)

print("="*50)
print("FEATURE CORRELATION WITH TARGET ('survived')")
print("="*50)
print(target_correlation)

# 5.3 Visualize Correlation Heatmap (Focus on top features)
# -------------------------------------------
# Select only the most relevant numeric columns for clarity
top_features = ['survived', 'sex_encoded', 'pclass', 'fare', 'age', 'Family_Size', 'sibsp', 'parch']
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Top Features with Target', fontsize=14)
plt.show()

# 5.4 Mutual Information (Wrapper/Filer Hybrid Method)
# -------------------------------------------
# Separate features (X) and target (y)
X = df.drop(columns=['survived'])  # All features except target
y = df['survived']  # Target

# Use Mutual Information to measure dependency between each feature and the target
selector = SelectKBest(score_func=mutual_info_classif, k='all')  # k='all' to get scores for all
selector.fit(X, y)

# Create a DataFrame to visualize feature importance based on Mutual Information
mi_scores = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': selector.scores_
}).sort_values(by='MI_Score', ascending=False)

print("="*50)
print("FEATURE IMPORTANCE (MUTUAL INFORMATION SCORES)")
print("="*50)
display(mi_scores)

# 5.5 Select Final Features for Modeling


final_features = ['sex_encoded', 'pclass', 'fare', 'age', 'Family_Size', 'port_Q', 'port_S']
X_final = df[final_features]
y_final = df['survived']

print("="*50)
print("FINAL SELECTED FEATURES FOR MODELING")
print("="*50)
print(f"Features: {final_features}")
print(f"Shape of Final Feature Set: {X_final.shape}")
print(f"Shape of Target: {y_final.shape}")

# 5.6 Train-Test Split (Final step to complete the pipeline)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

print("\nData successfully split for Machine Learning:")
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")
# ==========================================
# 6. ENCODE, SCALE, SPLIT, TRAIN, PREDICT & EVALUATE
# ==========================================

# 6.1 Import Modeling Libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import Multiple ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')  

print(" Modeling libraries imported successfully!")

# 6.2 Prepare Features (X) and Target (y)

feature_columns = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'embarked']
X = df[feature_columns].copy()  # Features
y = df['survived'].copy()       # Target

print("="*50)
print("DATA PREPARATION")
print("="*50)
print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Overall Survival Rate: {y.mean()*100:.2f}%")
print("\nFirst 5 rows of features:")
display(X.head())

# 6.3 Encode Categorical Data
# Encoding 'sex': Binary (0=Male, 1=Female)
X['sex'] = X['sex'].map({'male': 0, 'female': 1})


X = pd.get_dummies(X, columns=['embarked'], prefix='port', drop_first=True)

print("\n Categorical encoding complete. Features after encoding:")
display(X.head())

# -------------------------------------------
# 6.4 Split Training and Testing Data
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*50)
print("DATA SPLIT")
print("="*50)
print(f"Training set size: {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Training Survival Rate: {y_train.mean()*100:.2f}%")
print(f"Test Survival Rate: {y_test.mean()*100:.2f}%")

# 6.5 Scale Features (Standardization)

numeric_features = ['age', 'fare', 'sibsp', 'parch', 'pclass']  

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_features)  
    ],
    remainder='passthrough'  
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

feature_names = numeric_features + [col for col in X.columns if col not in numeric_features]
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

print("\n Feature scaling complete. Scaled feature names:")
print(feature_names)
print("\nFirst 5 rows of SCALED training data (notice age/fare are now normalized around 0):")
display(X_train_scaled.head()
# 6.6 Train Multiple ML Models & Evaluate

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# Store results for comparison
results = []

print("\n" + "="*60)
print("TRAINING & EVALUATING MULTIPLE MODELS")
print("="*60)

# Loop through each model
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🔹 MODEL: {name}")
    print('-'*50)
    

    model.fit(X_train_scaled, y_train)
    print(" Model trained successfully!")
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f" Performance on Test Set:")
    print(f"   - Accuracy  : {accuracy:.4f}  ({(accuracy*100):.2f}%)")
    print(f"   - Precision : {precision:.4f}")
    print(f"   - Recall    : {recall:.4f}")
    print(f"   - F1-Score  : {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n Confusion Matrix:")
    print(f"   [[TN: {cm[0,0]:3d}, FP: {cm[0,1]:3d}]")
    print(f"    [FN: {cm[1,0]:3d}, TP: {cm[1,1]:3d}]]")
    

# 6.7 Compare All Models Side-by-Side

print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

# Create a comparison DataFrame
comparison_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
display(comparison_df)

# Visualize the comparison
plt.figure(figsize=(12, 6))
comparison_melted = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=comparison_melted, x='Model', y='Score', hue='Metric')
plt.title('Model Performance Comparison (Accuracy, Precision, Recall, F1)', fontsize=14)
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 6.8 Select the Best Model & Interpret
# -------------------------------------------
best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
best_accuracy = comparison_df.iloc[0]['Accuracy']

print("="*60)
print(" BEST MODEL SELECTED")
print("="*60)
print(f" The best performing model is: '{best_model_name}'")
print(f"   - Test Accuracy: {best_accuracy*100:.2f}%")
print(f"   - F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")

print("\n Full Modeling Pipeline Complete! You are ready to deploy or fine-tune this model.")
# ==========================================
# 7. ADVANCED ML PIPELINE: CV, TUNING, SAVE/LOAD, IMPORTANCE, VIZ
# ==========================================

# -------------------------------------------
# 7.1 Import Required Advanced Libraries
# -------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, classification_report
from sklearn.tree import plot_tree
import joblib  # For saving and loading models
import warnings
warnings.filterwarnings('ignore')

# Ensure plots are rendered inline
%matplotlib inline

print(" Advanced libraries imported.")

# -------------------------------------------
# 7.2 Prepare Data (Starting from cleaned df)
# -------------------------------------------

X = df.drop(columns=['survived']).copy()
y = df['survived'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*50)
print("DATA SPLIT FOR PIPELINE")
print("="*50)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 7.3 Create a Full Preprocessing Pipeline

numeric_features = ['pclass', 'age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

# Numeric transformer: StandardScaler
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer: OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("\n Full pipeline created successfully!")
print(pipeline)

# 7.4 Perform Cross Validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='accuracy')

print("\n" + "="*50)
print("BASELINE CROSS-VALIDATION SCORES")
print("="*50)
print(f"Individual Fold Accuracies: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7.5 Hyperparameter Tuning with GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],        
    'classifier__max_depth': [None, 10, 20, 30],        
    'classifier__min_samples_split': [2, 5, 10],        
    'classifier__min_samples_leaf': [1, 2, 4]         
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,              
    scoring='accuracy',          
    n_jobs=-1,                  
    verbose=1                     
)

print("\n" + "="*50)
print("STARTING HYPERPARAMETER TUNING (GRID SEARCH)")
print("="*50)
print(f"Total combinations to test: {len(grid_search.param_grid)}")
print("This may take a minute...")

grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print("\n" + "="*50)
print("HYPERPARAMETER TUNING COMPLETE")
print("="*50)
print(f" Best Parameters: {best_params}")
print(f" Best Cross-Validation Accuracy: {best_cv_score:.4f}")

# Evaluate the tuned model on the UNSEEN test set
test_accuracy = best_model.score(X_test, y_test)
print(f" Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 7.6 Save and Load the Model

print("\n" + "="*50)
print("SAVING & LOADING MODEL")
print("="*50)

joblib.dump(best_model, 'titanic_best_model.pkl')
print(" Model saved as 'titanic_best_model.pkl'")

# Load the model back from the file
loaded_model = joblib.load('titanic_best_model.pkl')
print(" Model loaded successfully from disk!")

sample_predictions = loaded_model.predict(X_test.head())
print(f"Sample predictions on first 5 test rows: {sample_predictions}")
print(f"Actual labels: {y_test.head().values}")

# 7.7 Analyze Feature Importance

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

rf_model = best_model.named_steps['classifier']

preprocessor_fitted = best_model.named_steps['preprocessor']
numeric_feature_names = numeric_features

cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

all_feature_names = list(numeric_feature_names) + list(cat_feature_names)

importance_scores = rf_model.feature_importances_

# Create a DataFrame for easy viewing
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

display(importance_df)

# Print top 5 features
print("\n🔝 Top 5 Most Important Features:")
for idx, row in importance_df.head(5).iterrows():
    print(f"   - {row['Feature']}: {row['Importance']:.4f}")

# -------------------------------------------
# 7.8 Advanced Visualizations
# -------------------------------------------
print("\n" + "="*50)
print("GENERATING ADVANCED VISUALIZATIONS")
print("="*50)

# 7.8a Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 7.8b ROC Curve (Receiver Operating Characteristic)
# The ROC curve shows the trade-off between True Positive Rate and False Positive Rate
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f" Area Under the Curve (AUC): {roc_auc:.3f} — {'Excellent' if roc_auc > 0.9 else 'Good' if roc_auc > 0.8 else 'Moderate'}")

plt.figure(figsize=(20, 10))

first_tree = rf_model.estimators_[0]

plot_tree(first_tree, 
          feature_names=all_feature_names, 
          class_names=['Died', 'Survived'], 
          filled=True, 
          rounded=True, 
          max_depth=3,  
          fontsize=10)
plt.title('Visualization of the First Decision Tree in Random Forest (Depth=3)', fontsize=16)
plt.tight_layout()
plt.show()

# 7.8d Confusion Matrix Heatmap
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix - Tuned Random Forest (Test Set)', fontsize=14)
plt.grid(False)  
plt.show()

# Print detailed classification report
print("\n" + "="*50)
print("FINAL CLASSIFICATION REPORT (TUNED MODEL)")
print("="*50)
print(classification_report(y_test, best_model.predict(X_test), target_names=['Died', 'Survived']))

# FINAL SUMMARY

print("\n" + "="*60)
print(" FULL ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f" Cross-Validation Baseline: {cv_scores.mean():.4f}")
print(f" Best Tuned CV Score: {best_cv_score:.4f}")
print(f" Test Set Accuracy: {test_accuracy:.4f}")
print(f" ROC-AUC Score: {roc_auc:.3f}")
print(f" Model saved to: titanic_best_model.pkl")
print(" Plots displayed above.")
print("\n Next step: Try loading the model with 'joblib.load()' in a new session!")