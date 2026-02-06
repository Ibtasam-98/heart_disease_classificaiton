import pandas as pd
import numpy as np
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.feature_selection import SelectKBest, f_classif

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


def load_heart_disease_data():
    """Load the heart disease dataset"""
    df = pd.read_csv('dataset/heart.csv')
    return df


def basic_preprocessing(df):
    """Basic preprocessing: handle outliers"""
    df_clean = df.copy()
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

    return df_clean


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


def select_features(X, y, n_features=15):
    """Select best features using ANOVA F-value"""
    selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    return pd.DataFrame(X_selected, columns=selected_features), selected_features


def prepare_data(df):
    """Prepare data for training"""
    df_encoded = df.copy()

    # Encode categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # Feature selection
    X_selected, selected_features = select_features(X, y, n_features=15)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, selected_features


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)

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
        patience = 10

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
    print("HEART DISEASE CLASSIFICATION USING CNN")
    print("=" * 80)

    df = load_heart_disease_data()

    # Data
    data_summary = {
        'Total Samples': len(df),
        'Features': df.shape[1],
        'Positive Cases': df['target'].sum(),
        'Negative Cases': len(df) - df['target'].sum(),
        'Class Balance': f"{df['target'].mean():.2%}"
    }

    print_tabular("DATASET SUMMARY", data_summary)

    # Preprocessing
    df_clean = basic_preprocessing(df)
    df_engineered = feature_engineering(df_clean)

    # Prepare data
    X_train, X_test, y_train, y_test, selected_features = prepare_data(df_engineered)

    # Training data summary
    train_summary = pd.DataFrame({
        'Dataset': ['Training', 'Testing'],
        'Samples': [len(X_train), len(X_test)],
        'Features': [X_train.shape[1], X_test.shape[1]],
        'Positive Class': [y_train.sum(), y_test.sum()],
        'Negative Class': [len(y_train) - y_train.sum(), len(y_test) - y_test.sum()]
    })

    print_tabular("TRAINING/TESTING SPLIT", train_summary)

    # Selected features
    features_df = pd.DataFrame({
        'Feature Index': range(len(selected_features)),
        'Feature Name': selected_features
    })
    print_tabular("SELECTED FEATURES", features_df)

    # Train CNN model
    print_tabular("MODEL TRAINING", ["Training CNN model..."])

    start_time = time.time()
    cnn_model = CNNModel(input_size=X_train.shape[1])
    cnn_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on training set
    train_metrics, train_cm, train_report = evaluate_model(cnn_model, X_train, y_train, "Training Set")
    print_tabular("TRAINING SET PERFORMANCE", train_metrics)

    # Training confusion matrix
    if 'True Negatives' in train_cm:
        cm_train_df = pd.DataFrame(list(train_cm.items())[:4], columns=['Metric', 'Value'])
        print_tabular("TRAINING CONFUSION MATRIX", cm_train_df)

    # Evaluate on testing set
    test_metrics, test_cm, test_report = evaluate_model(cnn_model, X_test, y_test, "Testing Set")
    print_tabular("TESTING SET PERFORMANCE", test_metrics)

    # Testing confusion matrix
    if 'True Negatives' in test_cm:
        cm_test_df = pd.DataFrame(list(test_cm.items()), columns=['Metric', 'Value'])
        print_tabular("TESTING CONFUSION MATRIX", cm_test_df)

    # Classification report for test set
    report_df = pd.DataFrame(test_report).transpose().round(4)
    print_tabular("CLASSIFICATION REPORT", report_df)

    # Final summary
    train_acc = float(train_metrics['Accuracy'])
    test_acc = float(test_metrics['Accuracy'])

    summary_data = [
        ['Training Accuracy', train_metrics['Accuracy']],
        ['Testing Accuracy', test_metrics['Accuracy']],
        ['Accuracy Gap', f"{train_acc - test_acc:.4f}"],
        ['Overfitting Level', 'HIGH' if (train_acc - test_acc) > 0.05
        else 'MEDIUM' if (train_acc - test_acc) > 0.02
        else 'LOW'],
        ['Precision', test_metrics['Precision']],
        ['Recall', test_metrics['Recall']],
        ['F1-Score', test_metrics['F1-Score']],
        ['AUC-ROC', test_metrics.get('AUC-ROC', 'N/A')]
    ]

    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    print_tabular("FINAL MODEL SUMMARY", summary_df)

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()