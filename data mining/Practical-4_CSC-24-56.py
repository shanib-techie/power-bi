# classification_two_datasets_user_files.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Paths to your uploaded files
FRUIT_PATH = r"fruit_classification_dataset.csv"
BREAST_PATH = r"breast_cancer_classification_dataset.csv"


# Settings
RS_REPEATS = 5  # random subsampling repeats
RANDOM_STATE = 42

# Utility functions
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])


def evaluate_holdout(X, y, pipeline, train_size, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluate_random_subsampling(X, y, pipeline, train_size, repeats=RS_REPEATS):
    accs = []
    for i in range(repeats):
        rs = i * 7 + 1
        try:
            acc = evaluate_holdout(X, y, pipeline, train_size=train_size, random_state=rs)
            accs.append(acc)
        except Exception as e:
            # If something fails (rare), skip that repeat
            print(f"Random subsample repeat {i} failed: {e}")
    if len(accs) == 0:
        return np.nan, np.nan
    return np.mean(accs), np.std(accs)

def evaluate_cv(X, y, pipeline, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return np.mean(scores), np.std(scores)
    except Exception as e:
        print("Cross-validation failed:", e)
        return np.nan, np.nan


# Classifiers and splits
classifiers = {
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE)
}
train_sizes = {
    '75_25': 0.75,
    '66.6_33.3': 2/3
}

results = []


# Function to run full experiment on one dataset
def run_experiments(dataset_name, X, y):
    print(f"\n=== Running experiments for: {dataset_name} ===")
    preprocessor = build_preprocessor(X)
    for clf_name, clf in classifiers.items():
        for split_label, train_size in train_sizes.items():
            pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
            # Hold-out
            try:
                hold_acc = evaluate_holdout(X, y, pipeline, train_size=train_size, random_state=RANDOM_STATE)
            except Exception as e:
                print(f"Holdout failed for {dataset_name}, {clf_name}, {split_label}: {e}")
                hold_acc = np.nan
            # Random subsampling
            rs_mean, rs_std = evaluate_random_subsampling(X, y, pipeline, train_size=train_size, repeats=RS_REPEATS)
            # Cross-validation
            cv_mean, cv_std = evaluate_cv(X, y, pipeline, folds=5)

            results.append({
                'Dataset': dataset_name,
                'Classifier': clf_name,
                'Split': split_label,
                'Holdout_Accuracy': round(float(hold_acc), 4) if not pd.isna(hold_acc) else np.nan,
                'RandSub_mean': round(float(rs_mean), 4) if not pd.isna(rs_mean) else np.nan,
                'RandSub_std': round(float(rs_std), 4) if not pd.isna(rs_std) else np.nan,
                'CV_mean': round(float(cv_mean), 4) if not pd.isna(cv_mean) else np.nan,
                'CV_std': round(float(cv_std), 4) if not pd.isna(cv_std) else np.nan
            })
            print(f"{dataset_name} | {clf_name} | {split_label} -> Holdout: {hold_acc:.4f}, RandSub mean: {rs_mean:.4f}, CV mean: {cv_mean:.4f}")


# Dataset 1: Fruit dataset
print("Loading fruit dataset from:", FRUIT_PATH)
df_fruit = pd.read_csv(FRUIT_PATH)
df_fruit = df_fruit.dropna(axis=1, how='all')                       # drop fully-empty columns
df_fruit = df_fruit.loc[:, ~df_fruit.columns.str.startswith('Unnamed')]    # drop 'Unnamed' columns
    
# expected target: 'fruit_name'
TARGET_FRUIT = 'fruit_name'
if TARGET_FRUIT not in df_fruit.columns:
    raise ValueError(f"Target column '{TARGET_FRUIT}' not found in fruit dataset. Columns: {list(df_fruit.columns)}")

X_fruit = df_fruit.drop(columns=[TARGET_FRUIT]).copy()
y_fruit = df_fruit[TARGET_FRUIT].copy()
# factorize target if non-numeric
if y_fruit.dtype == 'object' or y_fruit.dtype.name == 'category':
    y_fruit, _ = pd.factorize(y_fruit)

run_experiments("Fruit", X_fruit, y_fruit)


# Dataset 2: Breast cancer dataset
print("\nLoading breast cancer dataset from:", BREAST_PATH)
df_breast = pd.read_csv(BREAST_PATH)
df_breast = df_breast.dropna(axis=1, how='all')                       # drop fully-empty columns
df_breast = df_breast.loc[:, ~df_breast.columns.str.startswith('Unnamed')]    # drop 'Unnamed' columns

# Common breast dataset columns: 'diagnosis' is typical target with 'M'/'B'
TARGET_BREAST = 'diagnosis'
if TARGET_BREAST not in df_breast.columns:
    raise ValueError(f"Target column '{TARGET_BREAST}' not found in breast dataset. Columns: {list(df_breast.columns)}")

# Drop id if present (non-feature)
if 'id' in df_breast.columns:
    df_breast = df_breast.drop(columns=['id'])

X_breast = df_breast.drop(columns=[TARGET_BREAST]).copy()
y_breast = df_breast[TARGET_BREAST].copy()
# factorize target to numeric labels (M/B -> 0/1)
if y_breast.dtype == 'object' or y_breast.dtype.name == 'category':
    y_breast, _ = pd.factorize(y_breast)

run_experiments("BreastCancer", X_breast, y_breast)


# Save & show summary
res_df = pd.DataFrame(results)
print("\n\n=== Final summary ===")
print(res_df)

CLASSIFCATION_RESULTS_CSV_PATH = "C:\\Users\\anshu\\OneDrive\\Desktop\\College\\DSE (Discipline Specific Elective Courses)\\Semester 3 - Data Mining I\\Practicals\\Datasets\\classification_results.csv"

res_df.to_csv(CLASSIFCATION_RESULTS_CSV_PATH, index=False)
print(f"\nSaved results to '{CLASSIFCATION_RESULTS_CSV_PATH}'")