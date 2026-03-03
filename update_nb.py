import nbformat as nbf
import os

# Path to the notebook
notebook_path = r'c:\Users\pc\Documents\AI\DL\projects\breast-cancer-detection\code.ipynb'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# We want to replace the cells that handle preprocessing and model building.
# Based on my view_file output:
# Cell 153: data.isnull().sum()
# Cell 154: categorical_data = ...
# Cell 155: encoder = OneHotEncoder...
# Cell 156: data.head()

# I will find cells by their content.
# Only replace once to avoid duplicate cells if run multiple times.

clean_replaced = False
train_replaced = False
redundant_replaced = False

new_cells = []

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'data.isnull().sum()' in cell.source and not clean_replaced:
             cell.source = """# Clean and Preprocess Data
if 'id' in data.columns or 'Unnamed: 32' in data.columns:
    data.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')

# Binary encoding for diagnosis (M=1, B=0)
if data['diagnosis'].dtype == 'object':
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

print("Data cleaned. Feature shape:", X.shape)"""
             clean_replaced = True
        elif 'encoder = OneHotEncoder' in cell.source and not train_replaced:
            cell.source = """from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Evaluate models
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{name} Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 30)"""
            train_replaced = True
        elif 'categorical_data = data.select_dtypes' in cell.source:
             cell.source = "# Redundant - categorical columns handled in cleaning step"
    new_cells.append(cell)

nb.cells = new_cells

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated successfully.")
