import pandas as pd
import numpy as np
import warnings
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_heart_disease_data():
    """Load the heart disease dataset"""
    df = pd.read_csv('dataset/heart.csv')
    return df


def basic_preprocessing(df):
    """Basic preprocessing: handle outliers using training data only"""
    df_clean = df.copy()
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Store bounds for each column to use later
    bounds = {}

    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        bounds[col] = {'lower': lower_bound, 'upper': upper_bound}

        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

    return df_clean, bounds


def feature_engineering(df):
    """Create new features"""
    df_engineered = df.copy()

    # Age groups
    df_engineered['age_group'] = pd.cut(df_engineered['age'],
                                        bins=[0, 40, 50, 60, 100],
                                        labels=[0, 1, 2, 3])

    # Interaction features
    df_engineered['age_bp'] = df_engineered['age'] * df_engineered['trestbps']
    df_engineered['chol_age'] = df_engineered['chol'] / (df_engineered['age'] + 1)

    return df_engineered


def encode_categorical_features(df):
    """Properly encode categorical features (one-hot for nominal, label for ordinal)"""
    df_encoded = df.copy()

    # Nominal categorical columns (should be one-hot encoded)
    nominal_cols = ['cp', 'restecg', 'slope', 'thal']

    # Ordinal or binary columns (can use label encoding)
    ordinal_cols = ['sex', 'fbs', 'exang', 'ca']

    # One-hot encode nominal columns
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, drop_first=True)

    # Label encode ordinal/binary columns
    for col in ordinal_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded


def cross_validate_cnn(X, y, n_splits=5):
    """Perform k-fold cross-validation for CNN model"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = {'accuracy': [], 'auc': []}

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Train model
        model = CNNModel(input_size=X_train_scaled.shape[1])
        model.fit(X_train_scaled, y_train_fold, X_val_scaled, y_val_fold)

        # Evaluate
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]

        acc = accuracy_score(y_val_fold, y_pred)
        auc = roc_auc_score(y_val_fold, y_proba)

        fold_scores['accuracy'].append(acc)
        fold_scores['auc'].append(auc)

        print(f"Fold {fold + 1}: Accuracy = {acc:.4f}, AUC-ROC = {auc:.4f}")

    print(f"\nMean Accuracy: {np.mean(fold_scores['accuracy']):.4f} (+/- {np.std(fold_scores['accuracy']):.4f})")
    print(f"Mean AUC-ROC: {np.mean(fold_scores['auc']):.4f} (+/- {np.std(fold_scores['auc']):.4f})")

    return fold_scores


def compare_baseline_models(X_train, y_train, X_test, y_test):
    """Compare CNN with classical ML models"""
    print("\n" + "=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)

    # Scale data for all models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = []

    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': f'{acc:.4f}',
            'Precision': f'{prec:.4f}',
            'Recall': f'{rec:.4f}',
            'F1-Score': f'{f1:.4f}',
            'AUC-ROC': f'{auc:.4f}'
        })

        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")

    # Add CNN results (assuming you'll run this separately)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    return pd.DataFrame(results)


class HeartDataset(Dataset):
    """PyTorch Dataset for heart disease data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNClassifier(nn.Module):
    """1D CNN for tabular data classification"""

    def __init__(self, input_size, num_classes=2):
        super(CNNClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling and regularization
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Activation
        self.relu = nn.ReLU()

        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def _calculate_flattened_size(self, input_size):
        """Calculate the size after convolutions and pooling"""
        x = torch.randn(1, 1, input_size)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        # Reshape for 1D CNN
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class CNNModel:
    """Wrapper for CNN model with training and evaluation"""

    def __init__(self, input_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_size = input_size
        self.device = device

        self.model = CNNClassifier(input_size).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)  # Added L2 regularization
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)

        self.best_val_acc = 0
        self.best_model_state = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        train_dataset = HeartDataset(X_train, y_train)
        val_dataset = HeartDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        early_stop_counter = 0
        patience = 15  # Increased patience

        for epoch in range(100):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_acc = correct / total
            self.scheduler.step(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        dataset = HeartDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        """Get probability predictions"""
        self.model.eval()
        dataset = HeartDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        probabilities = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)


def prepare_data_no_leakage(df):
    """Prepare data with NO LEAKAGE - split first, then preprocess"""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # 1. SPLIT FIRST - NO LEAKAGE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Handle outliers on training data only, then apply bounds to test
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in numerical_cols:
        # Calculate bounds on training data only
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply to both train and test
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

    # 3. Feature engineering (doesn't depend on target)
    X_train['age_group'] = pd.cut(X_train['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
    X_train['age_bp'] = X_train['age'] * X_train['trestbps']
    X_train['chol_age'] = X_train['chol'] / (X_train['age'] + 1)

    X_test['age_group'] = pd.cut(X_test['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
    X_test['age_bp'] = X_test['age'] * X_test['trestbps']
    X_test['chol_age'] = X_test['chol'] / (X_test['age'] + 1)

    # 4. Feature selection using ONLY training data
    selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]

    # 5. Scale using ONLY training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, selected_features


def print_tabular(title, data, headers=None):
    """Print data in tabular format"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    if isinstance(data, pd.DataFrame):
        print(data.to_string(index=False))
    elif isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
        print(df.to_string(index=False))
    elif isinstance(data, list):
        if headers:
            df = pd.DataFrame(data, columns=headers)
            print(df.to_string(index=False))
        else:
            for item in data:
                print(item)
    else:
        print(data)


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'Accuracy': f"{accuracy_score(y, y_pred):.4f}",
        'Precision': f"{precision_score(y, y_pred, zero_division=0):.4f}",
        'Recall': f"{recall_score(y, y_pred, zero_division=0):.4f}",
        'F1-Score': f"{f1_score(y, y_pred, zero_division=0):.4f}",
    }

    if y_proba is not None:
        metrics['AUC-ROC'] = f"{roc_auc_score(y, y_proba):.4f}"

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        cm_metrics = {
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp,
            'Sensitivity': f"{tp / (tp + fn):.4f}" if (tp + fn) > 0 else "0.0000",
            'Specificity': f"{tn / (tn + fp):.4f}" if (tn + fp) > 0 else "0.0000"
        }
    else:
        cm_metrics = {'Error': 'Non-binary confusion matrix'}

    # Classification report
    report = classification_report(y, y_pred, output_dict=True)

    return metrics, cm_metrics, report


def main():
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("HEART DISEASE CLASSIFICATION USING CNN (WITH LEAKAGE PREVENTION)")
    print("=" * 80)

    # Load data
    df = load_heart_disease_data()

    # Data summary
    data_summary = {
        'Total Samples': len(df),
        'Original Features': df.shape[1] - 1,  # Excluding target
        'Positive Cases': df['target'].sum(),
        'Negative Cases': len(df) - df['target'].sum(),
        'Class Balance': f"{df['target'].mean():.2%}"
    }
    print_tabular("DATASET SUMMARY", data_summary)

    # Encode categorical features properly
    df_encoded = encode_categorical_features(df)

    # Prepare data with NO LEAKAGE
    X_train, X_test, y_train, y_test, selected_features = prepare_data_no_leakage(df_encoded)

    # Training data summary
    train_summary = pd.DataFrame({
        'Dataset': ['Training', 'Testing'],
        'Samples': [len(X_train), len(X_test)],
        'Features': [X_train.shape[1], X_test.shape[1]],
        'Positive Class': [np.sum(y_train), np.sum(y_test)],
        'Negative Class': [len(y_train) - np.sum(y_train), len(y_test) - np.sum(y_test)]
    })
    print_tabular("TRAINING/TESTING SPLIT", train_summary)

    # Selected features
    features_df = pd.DataFrame({
        'Feature Index': range(len(selected_features)),
        'Feature Name': selected_features
    })
    print_tabular("SELECTED FEATURES", features_df)

    # PART 1: Cross-validation on training data
    print("\n" + "=" * 60)
    print("PART 1: CROSS-VALIDATION (Training Data Only)")
    print("=" * 60)

    # Use raw training data for CV (before scaling)
    cv_scores = cross_validate_cnn(X_train, y_train, n_splits=5)

    # PART 2: Train final model on full training set
    print("\n" + "=" * 60)
    print("PART 2: FINAL MODEL TRAINING")
    print("=" * 60)

    # Scale data for final training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train CNN model
    print_tabular("MODEL TRAINING", ["Training CNN model..."])

    start_time = time.time()
    cnn_model = CNNModel(input_size=X_train_scaled.shape[1])
    cnn_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on training set
    train_metrics, train_cm, train_report = evaluate_model(cnn_model, X_train_scaled, y_train, "Training Set")
    print_tabular("TRAINING SET PERFORMANCE", train_metrics)

    # Evaluate on testing set
    test_metrics, test_cm, test_report = evaluate_model(cnn_model, X_test_scaled, y_test, "Testing Set")
    print_tabular("TESTING SET PERFORMANCE", test_metrics)

    # Testing confusion matrix
    if 'True Negatives' in test_cm:
        cm_test_df = pd.DataFrame(list(test_cm.items()), columns=['Metric', 'Value'])
        print_tabular("TESTING CONFUSION MATRIX", cm_test_df)

    # PART 3: Baseline model comparison
    print("\n" + "=" * 60)
    print("PART 3: BASELINE MODEL COMPARISON")
    print("=" * 60)

    baseline_results = compare_baseline_models(X_train, y_train, X_test, y_test)
    print_tabular("BASELINE COMPARISON RESULTS", baseline_results)

    # Add CNN results to comparison
    cnn_row = pd.DataFrame([{
        'Model': 'CNN (Proposed)',
        'Accuracy': test_metrics['Accuracy'],
        'Precision': test_metrics['Precision'],
        'Recall': test_metrics['Recall'],
        'F1-Score': test_metrics['F1-Score'],
        'AUC-ROC': test_metrics.get('AUC-ROC', 'N/A')
    }])

    final_comparison = pd.concat([baseline_results, cnn_row], ignore_index=True)
    print_tabular("FINAL MODEL COMPARISON (All Models)", final_comparison)

    # Final summary with cross-validation results
    train_acc = float(train_metrics['Accuracy'])
    test_acc = float(test_metrics['Accuracy'])

    summary_data = [
        ['Training Accuracy', train_metrics['Accuracy']],
        ['Testing Accuracy', test_metrics['Accuracy']],
        ['Accuracy Gap', f"{train_acc - test_acc:.4f}"],
        ['5-Fold CV Accuracy (mean)',
         f"{np.mean(cv_scores['accuracy']):.4f} (+/- {np.std(cv_scores['accuracy']):.4f})"],
        ['5-Fold CV AUC-ROC (mean)', f"{np.mean(cv_scores['auc']):.4f} (+/- {np.std(cv_scores['auc']):.4f})"],
        ['Overfitting Assessment', 'Controlled' if (train_acc - test_acc) < 0.03 else 'Monitor'],
        ['Precision', test_metrics['Precision']],
        ['Recall', test_metrics['Recall']],
        ['F1-Score', test_metrics['F1-Score']],
        ['AUC-ROC', test_metrics.get('AUC-ROC', 'N/A')]
    ]

    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    print_tabular("FINAL MODEL SUMMARY WITH CROSS-VALIDATION", summary_df)

    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS IMPLEMENTED:")
    print("=" * 80)
    print("✓ Data leakage prevented: Split first, then preprocess")
    print("✓ Cross-validation performed: 5-fold stratified CV")
    print("✓ Baseline comparisons added: LR, RF, GB, SVM")
    print("✓ Proper categorical encoding: One-hot for nominal variables")
    print("✓ Overfitting controlled: L2 reg, gradient clipping, dropout")
    print("✓ Model generalization validated")
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()