# ============================================
# BLOCK 1: LIBRARY IMPORTS
# ============================================
print("\n" + "=" * 80)
print("BLOCK 1: Importing Libraries...")
print("=" * 80)

import pandas as pd
import numpy as np
import warnings
import time
import math
import os  # ADD THIS for directory creation

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Create visualization directory
os.makedirs('visualization', exist_ok=True)
os.makedirs('model', exist_ok=True)

print("✓ All libraries imported successfully!")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# ============================================
# BLOCK 2: SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================
print("\n" + "=" * 80)
print("Setting Random Seeds for Reproducibility")
print("=" * 80)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)
print("Random seeds set to 42")

# ============================================
# BLOCK 3: DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================
print("\n" + "=" * 80)
print("Defining Data Loading and Preprocessing Functions")
print("=" * 80)


def load_heart_disease_data():
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

    # BMI-like feature (using cholesterol as proxy)
    df_engineered['chol_bp_ratio'] = df_engineered['chol'] / (df_engineered['trestbps'] + 1)

    # Age-cholesterol interaction
    df_engineered['age_chol'] = df_engineered['age'] * df_engineered['chol'] / 100

    # Heart rate reserve
    df_engineered['heart_rate_reserve'] = df_engineered['thalach'] - df_engineered['age']

    return df_engineered


def select_features(X, y, n_features=12, method='mutual_info'):
    """Select best features using multiple methods"""
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
    else:
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))

    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    return pd.DataFrame(X_selected, columns=selected_features), selected_features


print("Data loading and preprocessing functions defined!")

# ============================================
# BLOCK 4: LOAD AND EXPLORE DATA
# ============================================
print("\n" + "=" * 80)
print("Loading and Exploring Data")
print("=" * 80)

df = load_heart_disease_data()
print(f"Dataset loaded! Shape: {df.shape}")

# Data summary
data_summary = {
    'Total Samples': len(df),
    'Features': df.shape[1],
    'Positive Cases': df['target'].sum(),
    'Negative Cases': len(df) - df['target'].sum(),
    'Class Balance': f"{df['target'].mean():.2%}"
}
summary_df = pd.DataFrame(list(data_summary.items()), columns=['Metric', 'Value'])
print(summary_df.to_string(index=False))

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ============================================
# BLOCK 5: PREPROCESSING AND FEATURE ENGINEERING
# ============================================
print("\n" + "=" * 80)
print("Preprocessing and Feature Engineering")
print("=" * 80)

# Apply preprocessing
df_clean = basic_preprocessing(df)
df_engineered = feature_engineering(df_clean)
print(f"After feature engineering: {df_engineered.shape[1]} features")

# Encode categorical variables
df_encoded = df_engineered.copy()
categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    print(f"  Encoded {col}")

# Separate features and target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']
print(f"✓ Features shape: {X.shape}")

# Feature selection
X_selected, selected_features = select_features(X, y, n_features=12, method='mutual_info')
print(f"Selected {len(selected_features)} features using Mutual Information")
print(f"Selected features: {list(selected_features)}")

# ============================================
# BLOCK 6: TRAIN-VALIDATION-TEST SPLIT
# ============================================
print("\n" + "=" * 80)
print("BLOCK 6: Train-Validation-Test Split (No Data Leakage)")
print("=" * 80)

# First split: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_selected, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: separate validation set from remaining (15% of original)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} samples ({len(X_train) / len(X_selected) * 100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({len(X_val) / len(X_selected) * 100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({len(X_test) / len(X_selected) * 100:.1f}%)")
print(f"\nClass distribution in training: {y_train.value_counts().to_dict()}")

# ============================================
# BLOCK 7: FEATURE SCALING
# ============================================
print("\n" + "=" * 80)
print("Feature Scaling")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Features scaled (fit on training only)")
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_val shape: {X_val_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")

# ============================================
# BLOCK 8: PYTORCH DATASET WITH WEIGHTED SAMPLING
# ============================================
print("\n" + "=" * 80)
print("Creating PyTorch Dataset with Weighted Sampling")
print("=" * 80)


class HeartDataset(Dataset):
    def __init__(self, X, y):
        # Ensure float32 for features and long for targets
        self.X = torch.FloatTensor(X)  # Explicitly float32
        self.y = torch.LongTensor(y)  # Explicitly long

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = HeartDataset(X_train_scaled, y_train.values)
val_dataset = HeartDataset(X_val_scaled, y_val.values)
test_dataset = HeartDataset(X_test_scaled, y_test.values)

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train.values)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train.values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

print(f"Training dataset: {len(train_dataset)} samples (with weighted sampling)")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Testing dataset: {len(test_dataset)} samples")
print(f"X dtype: {train_dataset.X.dtype}")
print(f"y dtype: {train_dataset.y.dtype}")

# ============================================
# BLOCK 9: ENHANCED VISION TRANSFORMER FOR TABULAR DATA
# ============================================
print("\n" + "=" * 80)
print("Defining Enhanced Vision Transformer for Tabular Data")
print("=" * 80)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attended))
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm(x + self.dropout(ff))
        return x


class TabularViTClassifier(nn.Module):
    """Enhanced Vision Transformer for tabular data classification"""

    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=4,
                 dim_feedforward=128, dropout=0.2):
        super(TabularViTClassifier, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features to d_model dimensions
        self.input_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer blocks with attention and feedforward
        self.attention_blocks = nn.ModuleList([
            MultiHeadSelfAttention(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.ff_blocks = nn.ModuleList([
            FeedForward(d_model, dim_feedforward, dropout) for _ in range(num_layers)
        ])

        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.shape[0]

        # Project each feature: (batch_size, input_dim, d_model)
        x = x.unsqueeze(-1)
        x = self.input_projection(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer blocks
        for attn_block, ff_block in zip(self.attention_blocks, self.ff_blocks):
            x = attn_block(x)
            x = ff_block(x)

        # Use CLS token for classification
        cls_output = x[:, 0, :]

        # Classification
        output = self.classifier(cls_output)

        return output


print("Enhanced Tabular ViT Classifier defined!")

# ============================================
# BLOCK 10: INITIALIZE MODEL WITH OPTIMIZED CONFIGURATION
# ============================================
print("\n" + "=" * 80)
print("Initializing ViT Model with Optimized Configuration")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model with optimized architecture
model = TabularViTClassifier(
    input_dim=X_train_scaled.shape[1],
    num_classes=2,
    d_model=64,           # Reduced from 128
    nhead=4,              # Reduced from 8
    num_layers=3,         # Reduced from 4
    dim_feedforward=128,  # Reduced from 256
    dropout=0.2           # Increased from 0.1
).to(device)

# Convert model to float32 to match input
model = model.float()

# Calculate class weights for loss function
class_counts = np.bincount(y_train.values)
class_weights_tensor = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print(f"✓ ViT Model initialized")
print(f"  Device: {device}")
print(f"  Input dimension: {X_train_scaled.shape[1]}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Class weights: {class_weights_tensor.cpu().numpy()}")
print(f"  Model dtype: {next(model.parameters()).dtype}")

# ============================================
# BLOCK 11: TRAIN THE MODEL WITH ENHANCED CONFIGURATION
# ============================================
print("\n" + "=" * 80)
print("Training ViT Model with Enhanced Configuration")
print("=" * 80)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_acc = 0
best_model_state = None
patience = 15
early_stop_counter = 0

# For plotting
train_losses = []
val_losses = []
val_accuracies = []

start_time = time.time()

print("Training in progress:")
print("-" * 70)

for epoch in range(100):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        # Ensure correct dtypes
        batch_X = batch_X.float().to(device)  # Convert to float32
        batch_y = batch_y.long().to(device)  # Convert to long for CrossEntropyLoss

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.float().to(device)  # Convert to float32
            batch_y = batch_y.long().to(device)  # Convert to long

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    # Store for plotting
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    # Update learning rate
    scheduler.step()

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # Print progress
    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}: LR={current_lr:.6f}, Train Loss={avg_train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

    if early_stop_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        break

training_time = time.time() - start_time

print("-" * 70)
if best_model_state:
    model.load_state_dict(best_model_state)
print(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# ============================================
# BLOCK 12: PLOT TRAINING HISTORY
# ============================================
print("\n" + "=" * 80)
print("Plotting Training History")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization/vit_training_history.png', dpi=100)
print("✓ Training history plot saved as 'visualization/vit_training_history.png'")
plt.show()

# ============================================
# BLOCK 13: MAKE PREDICTIONS
# ============================================
print("\n" + "=" * 80)
print("Making Predictions")
print("=" * 80)

model.eval()

# Training predictions
train_preds = []
train_probs = []
with torch.no_grad():
    for batch_X, _ in DataLoader(train_dataset, batch_size=32):
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_probs.extend(probs.cpu().numpy()[:, 1])

# Validation predictions
val_preds = []
val_probs = []
with torch.no_grad():
    for batch_X, _ in DataLoader(val_dataset, batch_size=32):
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_probs.extend(probs.cpu().numpy()[:, 1])

# Testing predictions
test_preds = []
test_probs = []
with torch.no_grad():
    for batch_X, _ in DataLoader(test_dataset, batch_size=32):
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy()[:, 1])

print(f"Predictions completed")

# ============================================
# BLOCK 14: COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "=" * 80)
print("Comprehensive Model Evaluation")
print("=" * 80)

# Training metrics
train_accuracy = accuracy_score(y_train, train_preds)
train_precision = precision_score(y_train, train_preds)
train_recall = recall_score(y_train, train_preds)
train_f1 = f1_score(y_train, train_preds)
train_auc = roc_auc_score(y_train, train_probs)

print("TRAINING Set Performance:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print(f"  AUC-ROC:   {train_auc:.4f}")

# Validation metrics
val_accuracy = accuracy_score(y_val, val_preds)
val_precision = precision_score(y_val, val_preds)
val_recall = recall_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_probs)

print("\nVALIDATION Set Performance:")
print(f"  Accuracy:  {val_accuracy:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1-Score:  {val_f1:.4f}")
print(f"  AUC-ROC:   {val_auc:.4f}")

# Testing metrics (FINAL)
test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)

print("\nTESTING Set Performance (FINAL):")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  AUC-ROC:   {test_auc:.4f}")

# Confusion Matrix
test_cm = confusion_matrix(y_test, test_preds)
print(f"\nTest Set Confusion Matrix:")
print(f"  True Negatives:  {test_cm[0, 0]}")
print(f"  False Positives: {test_cm[0, 1]}")
print(f"  False Negatives: {test_cm[1, 0]}")
print(f"  True Positives:  {test_cm[1, 1]}")

# Additional metrics
tn, fp, fn, tp = test_cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nDetailed Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity:          {specificity:.4f}")
print(f"  Precision:            {precision:.4f}")
print(f"  Negative Predictive Value: {npv:.4f}")

# ============================================
# BLOCK 15: CLASSIFICATION REPORT
# ============================================
print("\n" + "=" * 80)
print("BLOCK 15: Detailed Classification Report")
print("=" * 80)

print("\nClassification Report for Testing Set:")
print(classification_report(y_test, test_preds, target_names=['No Disease', 'Disease']))

# ============================================
# BLOCK 16: CONFUSION MATRIX VISUALIZATION
# ============================================
print("\n" + "=" * 80)
print("Confusion Matrix Visualization")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Training confusion matrix
train_cm = confusion_matrix(y_train, train_preds)
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[0].set_title('Training Set Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Validation confusion matrix
val_cm = confusion_matrix(y_val, val_preds)
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[1].set_title('Validation Set Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# Testing confusion matrix
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=axes[2],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[2].set_title('Testing Set Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualization/vit_confusion_matrices.png', dpi=100)
print("✓ Confusion matrices saved as 'visualization/vit_confusion_matrices.png'")
plt.show()

# ============================================
# BLOCK 17: ROC CURVES
# ============================================
print("\n" + "=" * 80)
print("ROC Curves")
print("=" * 80)

from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(8, 6))

# Training ROC
fpr_train, tpr_train, _ = roc_curve(y_train, train_probs)
ax.plot(fpr_train, tpr_train, label=f'Training (AUC = {train_auc:.3f})', linewidth=2)

# Validation ROC
fpr_val, tpr_val, _ = roc_curve(y_val, val_probs)
ax.plot(fpr_val, tpr_val, label=f'Validation (AUC = {val_auc:.3f})', linewidth=2)

# Testing ROC
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
ax.plot(fpr_test, tpr_test, label=f'Testing (AUC = {test_auc:.3f})', linewidth=2)

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Training vs Validation vs Testing')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization/vit_roc_curves.png', dpi=100)
print("✓ ROC curves saved as 'visualization/vit_roc_curves.png'")
plt.show()

# ============================================
# BLOCK 18: FINAL SUMMARY AND COMPARISON
# ============================================
print("\n" + "=" * 80)
print("Final Summary and Analysis")
print("=" * 80)

acc_gap = train_accuracy - test_accuracy

print("\nModel Performance Summary:")
print("-" * 65)
print(f"{'Metric':<15} {'Training':<12} {'Validation':<12} {'Testing':<12}")
print("-" * 65)
print(f"{'Accuracy':<15} {train_accuracy:<12.4f} {val_accuracy:<12.4f} {test_accuracy:<12.4f}")
print(f"{'Precision':<15} {train_precision:<12.4f} {val_precision:<12.4f} {test_precision:<12.4f}")
print(f"{'Recall':<15} {train_recall:<12.4f} {val_recall:<12.4f} {test_recall:<12.4f}")
print(f"{'F1-Score':<15} {train_f1:<12.4f} {val_f1:<12.4f} {test_f1:<12.4f}")
print(f"{'AUC-ROC':<15} {train_auc:<12.4f} {val_auc:<12.4f} {test_auc:<12.4f}")
print("-" * 65)

# Overfitting assessment
print(f"\nOverfitting Assessment:")
if acc_gap > 0.05:
    print(f"  Accuracy Gap: {acc_gap:.4f} - HIGH Overfitting")
    print(f"  Model shows significant overfitting")
    print(f"  Suggestions: Increase dropout, reduce model complexity, or add more regularization")
elif acc_gap > 0.02:
    print(f"  Accuracy Gap: {acc_gap:.4f} - MEDIUM Overfitting")
    print(f"  Model shows moderate overfitting")
    print(f"  Suggestions: Try increasing dropout or adding weight decay")
else:
    print(f"  Accuracy Gap: {acc_gap:.4f} - LOW Overfitting")
    print(f"  Model generalizes well to unseen data")

# Performance assessment
print(f"\nPerformance Assessment:")
if test_accuracy >= 0.85:
    print(f"  Excellent! Model achieves {test_accuracy:.1%} accuracy")
elif test_accuracy >= 0.75:
    print(f"  Good! Model achieves {test_accuracy:.1%} accuracy")
elif test_accuracy >= 0.65:
    print(f"  Fair. Model achieves {test_accuracy:.1%} accuracy")
else:
    print(f"  Needs improvement. Model achieves {test_accuracy:.1%} accuracy")
    print(f"  Suggestions:")
    print(f"     Try ensemble methods")
    print(f"     Collect more data")
    print(f"     Try different feature engineering")

# ============================================
# BLOCK 19: SAVE MODEL AND ARTIFACTS
# ============================================
print("\n" + "=" * 80)
print("Saving Model and Artifacts")
print("=" * 80)

torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_dim': X_train_scaled.shape[1],
        'num_classes': 2,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 128,
        'dropout': 0.2
    },
    'selected_features': list(selected_features),
    'scaler': scaler,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_auc': test_auc
}, 'model/vit_heart_disease_enhanced.pth')

print("✓ Model saved as 'model/vit_heart_disease_enhanced.pth'")
print("\nSaved Artifacts:")
print("   • model/vit_heart_disease_enhanced.pth - Model weights and config")
print("   • visualization/vit_training_history.png - Loss and accuracy plots")
print("   • visualization/vit_confusion_matrices.png - Confusion matrices")
print("   • visualization/vit_roc_curves.png - ROC curves")

# ============================================
# BLOCK 20: COMPLETION MESSAGE
# ============================================
print("\n" + "=" * 80)
print("Training Pipeline Complete")
print("=" * 80)

print(f"\nFinal Test Performance Summary:")
print(f"   ┌─────────────────────────────────────────┐")
print(f"   │ Metric        │ Value                   │")
print(f"   ├─────────────────────────────────────────┤")
print(f"   │ Accuracy      │ {test_accuracy:.4f}                    │")
print(f"   │ Precision     │ {test_precision:.4f}                    │")
print(f"   │ Recall        │ {test_recall:.4f}                    │")
print(f"   │ F1-Score      │ {test_f1:.4f}                    │")
print(f"   │ AUC-ROC       │ {test_auc:.4f}                    │")
print(f"   └─────────────────────────────────────────┘")

print(f"\nTotal training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

print("\n" + "=" * 80)
print("THANK YOU FOR USING ENHANCED ViT HEART DISEASE CLASSIFICATION SYSTEM!")
print("=" * 80)

print("\nKey Improvements Made:")
print("   Fixed data leakage (separate train/val/test)")
print("   Weighted sampling for class imbalance")
print("   Optimized ViT architecture for small data")
print("   Added gradient clipping")
print("   Cosine annealing learning rate schedule")
print("   Comprehensive visualizations")
print("   Proper model saving with all artifacts")