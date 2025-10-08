1 — End-to-end supervised pipeline (Heart Disease classification)

# heart_pipeline.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 1) Load dataset (assumes heart.csv in working dir)
df = pd.read_csv("heart.csv")  # adapt filename/columns as needed
print("Shape:", df.shape)
print(df.head())

# 2) Quick data inspection
print(df.info())
print(df.isna().sum())

# Example: assume 'target' is the binary label (1 = disease, 0 = no disease)
TARGET = "target"

# 3) Split features into numeric and categorical (adjust names to your dataset)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET).tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

# 4) Train/test split (stratified to preserve class ratios)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Preprocessing pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # robust to outliers
    ("scaler", StandardScaler())                     # many models need scaled features
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, cat_cols)
])

# 6) Baseline model: Logistic Regression
pipe_lr = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipe_rf = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 7) Train baseline models
pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# 8) Evaluate utilities
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:    ", acc)
    print("Precision:   ", prec)
    print("Recall:      ", rec)
    print("F1-score:    ", f1)
    if roc is not None:
        print("ROC AUC:     ", roc)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred, zero_division=0))

print("\nLogistic Regression performance on test set:")
evaluate_model(pipe_lr, X_test, y_test)

print("\nRandom Forest performance on test set:")
evaluate_model(pipe_rf, X_test, y_test)

# 9) Hyperparameter tuning (example for Random Forest)
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_split": [2, 5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_rf = GridSearchCV(pipe_rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)
print("Best RF CV ROC AUC:", grid_rf.best_score_)

# Final evaluation
best_rf = grid_rf.best_estimator_
print("\nBest Random Forest on test set:")
evaluate_model(best_rf, X_test, y_test)

# 10) Feature importance (if using tree-based model)
# We need feature names after preprocessing
# Build feature name list:
preproc = best_rf.named_steps["preprocess"]
# numeric feature names
num_feats = numeric_cols
# categorical onehot names
if cat_cols:
    # get encoder categories
    ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
else:
    cat_names = []

feature_names = num_feats + cat_names

importances = best_rf.named_steps["clf"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop feature importances:\n", fi.head(10))


2.— Fake news classification (text) & model comparison

# fake_news_pipeline.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

# 1) Load dataset (assumes 'fake_or_real_news.csv' or similar)
# The dataset should have columns: 'text' and 'label' (label: 'FAKE'/'REAL' or 0/1)
df = pd.read_csv("fake_or_real_news.csv")
print(df.shape)
df = df.dropna(subset=["text", "label"])
X = df["text"]
y = df["label"].map(lambda v: 1 if str(v).lower() in ["fake", "1", "true"] else 0)  # convert to 1=fake,0=real

# 2) Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Basic TF-IDF + classifier pipelines
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),   # unigrams + bigrams often help
    max_df=0.95,
    min_df=5
)

# Logistic Regression pipeline
pipe_lr = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(max_iter=1000, solver="saga", penalty="l2", random_state=42))
])

# Decision Tree pipeline
pipe_dt = Pipeline([
    ("tfidf", tfidf),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# SVM pipeline (LinearSVC; wrap in CalibratedClassifierCV to get probabilities)
pipe_svm = Pipeline([
    ("tfidf", tfidf),
    ("clf", CalibratedClassifierCV(LinearSVC(random_state=42), cv=3))  # calibrated for proba
])

# 4) Fit models (be mindful of runtime; for large data consider sampling or increasing min_df)
print("Training Logistic Regression...")
pipe_lr.fit(X_train, y_train)
print("Training Decision Tree...")
pipe_dt.fit(X_train, y_train)
print("Training SVM (may take longer)...")
pipe_svm.fit(X_train, y_train)

# 5) Evaluate helper
def eval_and_report(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    # predict_proba may not exist; use decision_function / calibrated clf
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    print(f"\n=== {name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    if y_proba is not None:
        try:
            print("ROC AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

eval_and_report(pipe_lr, X_test, y_test, "Logistic Regression")
eval_and_report(pipe_dt, X_test, y_test, "Decision Tree")
eval_and_report(pipe_svm, X_test, y_test, "SVM (Linear)")

# 6) Quick grid search example (for logistic regression)
param_grid_lr = {
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gs = GridSearchCV(pipe_lr, param_grid_lr, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params for LR:", gs.best_params_)
eval_and_report(gs.best_estimator_, X_test, y_test, "Tuned Logistic Regression")


3. Ridge and Lasso Regression on a Multicollinear Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(42)

# Generate features
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = X1 + np.random.randn(n_samples) * 0.1  # highly correlated with X1
X3 = np.random.randn(n_samples)
X = np.column_stack([X1, X2, X3])

# Target variable
y = 3*X1 + 2*X2 + 1.5*X3 + np.random.randn(n_samples) * 0.5

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Step 2: Train Ridge and Lasso Regression

# Ridge regression
ridge = Ridge(alpha=1.0)  # alpha = regularization strength
ridge.fit(X_train_scaled, y_train)

# Lasso regression
lasso = Lasso(alpha=0.1)  # alpha = regularization strength
lasso.fit(X_train_scaled, y_train)

print("Ridge coefficients:", ridge.coef_)
print("Lasso coefficients:", lasso.coef_)

Step 3: Analyze effect of regularization via coefficient plots

coef_df = pd.DataFrame({
    "Feature": ["X1", "X2", "X3"],
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

coef_df.set_index("Feature").plot(kind="bar", figsize=(8,5))
plt.title("Effect of Regularization on Coefficients")
plt.ylabel("Coefficient Value")
plt.show()Step 3: Analyze effect of regularization via coefficient plots
coef_df = pd.DataFrame({
    "Feature": ["X1", "X2", "X3"],
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

coef_df.set_index("Feature").plot(kind="bar", figsize=(8,5))
plt.title("Effect of Regularization on Coefficients")
plt.ylabel("Coefficient Value")
plt.show()


4. Logistic Regression with L1 and L2 Regularization

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create dataset with some redundant features
X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                           n_redundant=3, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


Step 2: Train Logistic Regression with L1 and L2

# L1 regularization
logreg_l1 = LogisticRegression(penalty="l1", solver="saga", max_iter=5000)
logreg_l1.fit(X_train_scaled, y_train)

# L2 regularization
logreg_l2 = LogisticRegression(penalty="l2", solver="saga", max_iter=5000)
logreg_l2.fit(X_train_scaled, y_train)

# Predictions
y_pred_l1 = logreg_l1.predict(X_test_scaled)
y_pred_l2 = logreg_l2.predict(X_test_scaled)

# Accuracy
acc_l1 = accuracy_score(y_test, y_pred_l1)
acc_l2 = accuracy_score(y_test, y_pred_l2)

print("Accuracy L1:", acc_l1)
print("Accuracy L2:", acc_l2)

# Sparsity (number of zero coefficients)
sparsity_l1 = np.sum(logreg_l1.coef_ == 0)
sparsity_l2 = np.sum(logreg_l2.coef_ == 0)
print("Sparsity L1 (zero coefficients):", sparsity_l1)
print("Sparsity L2 (zero coefficients):", sparsity_l2)


Step 3: Compare coefficients visually

coef_df = pd.DataFrame({
    "Feature": [f"X{i+1}" for i in range(X.shape[1])],
    "L1": logreg_l1.coef_.flatten(),
    "L2": logreg_l2.coef_.flatten()
})

coef_df.set_index("Feature").plot(kind="bar", figsize=(10,5))
plt.title("Logistic Regression Coefficients: L1 vs L2")
plt.ylabel("Coefficient Value")
plt.show()


5. Feature Selection: Backward Elimination vs Lasso Regression

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant for intercept
X_train_sm = sm.add_constant(X_train)
X_opt = X_train_sm.copy()

# Perform backward elimination
while True:
    model = sm.OLS(y_train, X_opt).fit()
    p_values = model.pvalues
    max_pval = p_values.drop("const").max()  # ignore intercept
    if max_pval > 0.05:  # significance level
        excluded_feature = p_values.drop("const").idxmax()
        X_opt = X_opt.drop(columns=[excluded_feature])
        print(f"Dropping feature: {excluded_feature} with p-value {max_pval:.4f}")
    else:
        break

print("\nSelected features after backward elimination:")
print(X_opt.columns)


from sklearn.preprocessing import StandardScaler

# Scale features for Lasso
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso regression
lasso = Lasso(alpha=0.1)  # alpha = regularization strength
lasso.fit(X_train_scaled, y_train)

# Check non-zero coefficients
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lasso.coef_
})
selected_features_lasso = coef_df[coef_df["Coefficient"] != 0]
print("\nFeatures selected by Lasso:")
print(selected_features_lasso)


6. Train a Decision Tree on a Medical Dataset and Prune It

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Diabetes regression dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full tree (no pruning)
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

# Plot tree
plt.figure(figsize=(20,8))
plot_tree(tree, feature_names=X.columns, filled=True, max_depth=2)
plt.title("Decision Tree (first 2 levels shown)")
plt.show()

# Evaluate
from sklearn.metrics import mean_squared_error
y_pred = tree.predict(X_test)
print("MSE (full tree):", mean_squared_error(y_test, y_pred))

# Depth-limited tree
tree_depth = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_depth.fit(X_train, y_train)

y_pred_depth = tree_depth.predict(X_test)
print("MSE (max_depth=4):", mean_squared_error(y_test, y_pred_depth))


# Get effective alphas for pruning
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train trees for different alphas
trees = []
for ccp_alpha in ccp_alphas:
    t = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    t.fit(X_train, y_train)
    trees.append(t)

# Plot MSE vs alpha
train_mse = [mean_squared_error(y_train, t.predict(X_train)) for t in trees]
test_mse = [mean_squared_error(y_test, t.predict(X_test)) for t in trees]

plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, train_mse, marker='o', label='Train MSE')
plt.plot(ccp_alphas, test_mse, marker='o', label='Test MSE')
plt.xlabel("ccp_alpha")
plt.ylabel("MSE")
plt.title("Cost Complexity Pruning")
plt.legend()
plt.show()

# Choose optimal alpha (e.g., lowest test MSE)
best_alpha = ccp_alphas[np.argmin(test_mse)]
tree_pruned = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
tree_pruned.fit(X_train, y_train)
print("Best ccp_alpha:", best_alpha)
print("MSE (pruned tree):", mean_squared_error(y_test, tree_pruned.predict(X_test)))


7. Feature Importance in Random Forest

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color="skyblue")
plt.gca().invert_yaxis()  # largest on top
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

print("Top 5 features:\n", feat_imp_df.head(5))


8. Ensemble Voting Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                           n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for Logistic Regression and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Individual classifiers
clf1 = LogisticRegression(random_state=42, max_iter=1000)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(kernel='linear', probability=True, random_state=42)

# Fit individual models
clf1.fit(X_train_scaled, y_train)
clf2.fit(X_train, y_train)  # Decision Tree works fine without scaling
clf3.fit(X_train_scaled, y_train)

# Predictions & performance of individual models
print("Logistic Regression Accuracy:", accuracy_score(y_test, clf1.predict(X_test_scaled)))
print("Decision Tree Accuracy:", accuracy_score(y_test, clf2.predict(X_test)))
print("SVM Accuracy:", accuracy_score(y_test, clf3.predict(X_test_scaled)))

# Ensemble Voting Classifier (soft voting)
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)],
    voting='soft'  # use probabilities
)
voting_clf.fit(X_train_scaled, y_train)  # all input must be scaled
y_pred_ensemble = voting_clf.predict(X_test_scaled)

# Ensemble performance
print("\nVoting Ensemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("\nClassification Report:\n", classification_report(y_test, y_pred_ensemble))


9. Compare Bagging vs Boosting with Multiple Base Learners

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=6, n_redundant=2,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for KNN and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base learners
base_learners = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Evaluate Bagging and AdaBoost for each base learner
for name, base in base_learners.items():
    print(f"\n--- Base Learner: {name} ---")
    
    # Bagging
    bag = BaggingClassifier(base_estimator=base, n_estimators=50, random_state=42)
    if name in ["KNN", "Logistic Regression"]:
        bag.fit(X_train_scaled, y_train)
        y_pred_bag = bag.predict(X_test_scaled)
    else:
        bag.fit(X_train, y_train)
        y_pred_bag = bag.predict(X_test)
    print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bag))
    
    # Boosting (AdaBoost)
    if name == "Decision Tree":
        ada = AdaBoostClassifier(base_estimator=base, n_estimators=50, learning_rate=1.0, random_state=42)
        ada.fit(X_train, y_train)
        y_pred_ada = ada.predict(X_test)
        print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
    else:
        print("AdaBoost not typically used with KNN or Logistic Regression base learners (only trees)")



10. Analyze Effect of n_estimators and learning_rate on AdaBoost

from sklearn.model_selection import GridSearchCV

# Decision Tree as base learner
base = DecisionTreeClassifier(max_depth=1, random_state=42)

# AdaBoost pipeline
ada = AdaBoostClassifier(base_estimator=base, random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.5, 1.0]
}

grid = GridSearchCV(ada, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)

# Evaluate on test set
best_ada = grid.best_estimator_
y_pred_test = best_ada.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred_test))


11. Stacking Classifier with Logistic Regression as Meta-Learner

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1) Create dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                           n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) Scale features for LR and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3) Base learners
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# 4) Stacking classifier with Logistic Regression as meta-learner
stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
stack.fit(X_train_scaled, y_train)
y_pred_stack = stack.predict(X_test_scaled)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stack))

# 5) Compare with Bagging (Decision Tree)
bag = BaggingClassifier(DecisionTreeClassifier(max_depth=3, random_state=42), n_estimators=50, random_state=42)
bag.fit(X_train_scaled, y_train)
y_pred_bag = bag.predict(X_test_scaled)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bag))

# 6) Compare with AdaBoost (Decision Tree)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42),
                         n_estimators=50, learning_rate=1.0, random_state=42)
ada.fit(X_train_scaled, y_train)
y_pred_ada = ada.predict(X_test_scaled)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))


12. SVM with Different Kernels & Decision Boundary Visualization

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
import numpy as np

# 1) Create 2D classification dataset for visualization
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 2) Define a function to plot decision boundaries
def plot_svm_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.show()

# 3) Train SVMs with different kernels
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    if kernel == 'poly':
        model = SVC(kernel='poly', degree=3, C=1.0)
    else:
        model = SVC(kernel=kernel, C=1.0)
    model.fit(X, y)
    plot_svm_boundary(model, X, y, f"SVM with {kernel} kernel")

# 4) Grid Search for best kernel parameters
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'degree': [2,3,4],   # only used for poly
    'gamma': ['scale', 'auto']  # only for rbf/poly
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print("Best SVM parameters:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)


13. Multi-Class Classification with One-vs-One and One-vs-Rest SVM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-vs-Rest (OvR)
svm_ovr = SVC(decision_function_shape='ovr', kernel='linear', random_state=42)
svm_ovr.fit(X_train_scaled, y_train)
y_pred_ovr = svm_ovr.predict(X_test_scaled)
print("OvR Accuracy:", accuracy_score(y_test, y_pred_ovr))

# Confusion matrix OvR
cm_ovr = confusion_matrix(y_test, y_pred_ovr)
ConfusionMatrixDisplay(cm_ovr, display_labels=iris.target_names).plot()
plt.title("One-vs-Rest SVM Confusion Matrix")
plt.show()

# One-vs-One (OvO)
svm_ovo = SVC(decision_function_shape='ovo', kernel='linear', random_state=42)
svm_ovo.fit(X_train_scaled, y_train)
y_pred_ovo = svm_ovo.predict(X_test_scaled)
print("OvO Accuracy:", accuracy_score(y_test, y_pred_ovo))

# Confusion matrix OvO
cm_ovo = confusion_matrix(y_test, y_pred_ovo)
ConfusionMatrixDisplay(cm_ovo, display_labels=iris.target_names).plot()
plt.title("One-vs-One SVM Confusion Matrix")
plt.show()


14. SVM for High-Dimensional Text Classification (e.g., Spam Filtering)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example text dataset
texts = [
    "Win money now", "Hello friend", "Limited offer, click here", "Meeting at 10am",
    "You won a prize", "Lunch today?", "Free tickets available", "Are you coming?"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM classifier
svm_text = LinearSVC()
svm_text.fit(X_train_tfidf, y_train)
y_pred = svm_text.predict(X_test_tfidf)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Advantages of SVM for high-dimensional text:
# - Handles sparse data efficiently (TF-IDF produces sparse matrices)
# - Maximizes margin → better generalization
# - Linear SVM often sufficient for text classification


15. Create a confusion matrix and classification report for a multi-class classifier. Interpret results
and suggest ways to improve precision.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# 1) Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Train multi-class SVM classifier
svm = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

# 5) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap=plt.cm.Blues)
plt.title("Multi-Class Confusion Matrix")
plt.show()

# 6) Classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification Report:\n", report)

# 7) Overall accuracy
print("Overall Accuracy:", accuracy_score(y_test, y_pred))


16. k-Fold Cross-Validation for Multiple Supervised Models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Stratified k-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation scores
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name} CV Accuracy: Mean={scores.mean():.3f}, Std={scores.std():.3f}")

# Visualize variation
plt.figure(figsize=(8,5))
plt.boxplot(cv_results.values(), labels=cv_results.keys())
plt.ylabel("Accuracy")
plt.title("k-Fold Cross-Validation Performance")
plt.show()


17. Hyperparameter Tuning Using GridSearchCV and RandomizedSearchCV

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Split dataset (just for demonstration, can use full data with CV)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)
print("RF CV Accuracy:", grid_rf.best_score_)

# ---------------- SVM ----------------
svm = SVC(random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)
print("Best SVM params:", grid_svm.best_params_)
print("SVM CV Accuracy:", grid_svm.best_score_)

# ---------------- Gradient Boosting ----------------
gb = GradientBoostingClassifier(random_state=42)
param_dist_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1.0]
}

random_gb = RandomizedSearchCV(gb, param_distributions=param_dist_gb, n_iter=10, cv=5,
                               scoring='accuracy', n_jobs=-1, random_state=42)
random_gb.fit(X_train, y_train)
print("Best GB params:", random_gb.best_params_)
print("GB CV Accuracy:", random_gb.best_score_)


18. Employee Attrition Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Example synthetic employee dataset
data = pd.DataFrame({
    'Age': [25, 30, 45, 28, 40, 35, 50, 29],
    'Salary': [50000, 60000, 90000, 52000, 85000, 70000, 95000, 58000],
    'YearsAtCompany': [1, 5, 10, 2, 8, 6, 12, 3],
    'JobSatisfaction': [3, 2, 4, 3, 5, 4, 2, 3],
    'Attrition': [1, 0, 0, 1, 0, 0, 0, 1]  # 1 = leaving, 0 = staying
})

X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf.predict(X_test_scaled)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Interpretability: Feature Importance
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
plt.title("Feature Importance for Attrition Prediction")
plt.show()


19. Fraud Transaction Detection (Imbalanced Dataset)

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Synthetic imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=2,
                           weights=[0.95, 0.05], flip_y=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to oversample minority class
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Precision-Recall Curve
y_scores = rf.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, marker='.', label=f'PR AUC={pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


20. Real-Time Sentiment Classification using Streamlit

# Save this as app.py and run: streamlit run app.py
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data
texts = ["I love this product", "Terrible service", "Great quality", "Not worth it", "Excellent experience", "Very bad"]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Streamlit interface
st.title("Sentiment Classifier")
user_input = st.text_input("Enter your review:")

if st.button("Predict"):
    if user_input:
        user_vect = vectorizer.transform([user_input])
        prediction = model.predict(user_vect)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review.")

