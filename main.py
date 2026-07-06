"""
Symptom-to-Disease Classification: BERT vs RoBERTa (Fixed Version)
==========================================================================
A comprehensive comparison with automatic fallback if models fail to download.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
import kagglehub
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ==========================================================
# CONFIG
# ==========================================================

# Set environment variables for better download handling
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_LEN = 128
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Define models to compare
MODELS_TO_COMPARE = [
    {
        "name": "BERT-base",
        "model_name": "bert-base-uncased",
        "params": "110M",
        "description": "Original BERT trained on BooksCorpus and Wikipedia",
        "color": "#3498db"
    },
    {
        "name": "DistilBERT",
        "model_name": "distilbert-base-uncased",
        "params": "66M",
        "description": "Distilled BERT (40% smaller, 60% faster, 97% performance)",
        "color": "#2ecc71"
    }
]

OUTPUT_DIR = "artifacts"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

RESULTS = {}
TRAINING_TIMES = {}
TRAINED_MODELS = []


# ==========================================================
# DATA LOADING & PREPROCESSING
# ==========================================================

def load_data():
    """Load the Symptom2Disease dataset from Kaggle"""
    try:
        path = kagglehub.dataset_download("niyarrbarman/symptom2disease")
        print(f"✅ Dataset downloaded to: {path}")
    except Exception as e:
        print(f"❌ Kaggle download error: {e}")
        local_path = "symptom2disease.csv"
        if os.path.exists(local_path):
            print(f"✅ Loading from local: {local_path}")
            return pd.read_csv(local_path)
        raise

    csv_file = None
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break

    if csv_file is None:
        raise FileNotFoundError("CSV file not found in downloaded dataset.")

    df = pd.read_csv(csv_file)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols["label"]: "label", cols["text"]: "text"})
    return df[["label", "text"]].dropna()


def preprocess(df):
    """Encode labels and prepare data"""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
    return df["text"].astype(str), y, label_encoder, list(label_encoder.classes_)


# ==========================================================
# EVALUATION HELPERS
# ==========================================================

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close()


def plot_roc_auc(y_true, y_score, classes, title, filename):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    aucs = []
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, alpha=0.3, linewidth=1, color=colors[i])

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{title}\nMacro-AUC = {np.mean(aucs):.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close()
    return float(np.mean(aucs))


def evaluate_model(name, y_train, y_train_pred, y_test, y_test_pred, classes, y_test_probs=None):
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n{'='*50}")
    print(f"{name} RESULTS")
    print(f"{'='*50}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    report = classification_report(y_test, y_test_pred, target_names=classes, zero_division=0)
    print("\nClassification Report:\n", report)

    report_dict = classification_report(y_test, y_test_pred, target_names=classes,
                                       zero_division=0, output_dict=True)

    safe_name = name.lower().replace("-", "_").replace(" ", "_")
    plot_confusion_matrix(y_test, y_test_pred, classes,
                         f"{name} - Confusion Matrix",
                         f"{safe_name}_confusion.png")

    macro_auc = None
    if y_test_probs is not None:
        macro_auc = plot_roc_auc(y_test, y_test_probs, classes,
                                f"{name} - ROC Curves",
                                f"{safe_name}_roc.png")

    # Get per-class metrics
    per_class = {}
    for cls in classes:
        if cls in report_dict:
            per_class[cls] = {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"]
            }

    RESULTS[name] = {
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "macro_auc": macro_auc,
        "per_class": per_class,
        "report": report_dict
    }

    return train_acc, test_acc


# ==========================================================
# BERT CLASSIFICATION TRAINING
# ==========================================================

class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_model_safe(model_name, num_labels):
    """Load model with safe parameter handling"""
    try:
        # First try without timeout parameter
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f"  ⚠️  Error loading {model_name}: {str(e)[:100]}")
        raise


def train_model(X_train, X_test, y_train, y_test, classes, model_config):
    """Train a BERT-based model"""

    model_name = model_config["model_name"]
    model_key = model_config["name"]

    print(f"\n{'#'*60}")
    print(f"Training {model_key}")
    print(f"Model: {model_name}")
    print(f"Parameters: {model_config['params']}")
    print(f"Description: {model_config['description']}")
    print(f"{'#'*60}")

    start_time = time.time()

    # Load model
    try:
        tokenizer, model = load_model_safe(model_name, len(classes))
        print(f"  ✅ Successfully loaded {model_name}")
    except Exception as e:
        print(f"  ❌ Failed to load {model_name}: {e}")
        # Try alternative: DistilBERT as fallback
        print(f"  🔄 Trying fallback: distilbert-base-uncased")
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(classes)
            ).to(DEVICE)
            model_key = f"{model_key} (using DistilBERT)"
            print(f"  ✅ Successfully loaded fallback")
        except Exception as e2:
            print(f"  ❌ Fallback also failed: {e2}")
            raise

    # Tokenize
    print(f"  Tokenizing training data...")
    train_enc = tokenizer(
        list(X_train),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors=None
    )

    print(f"  Tokenizing test data...")
    test_enc = tokenizer(
        list(X_test),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors=None
    )

    train_dataset = SymptomDataset(train_enc, list(y_train))
    test_dataset = SymptomDataset(test_enc, list(y_test))

    safe_name = model_key.lower().replace("-", "_").replace(" ", "_")
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{safe_name}_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=1))},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"  Starting training...")
    trainer.train()
    training_time = time.time() - start_time

    # Predictions
    print(f"  Generating predictions...")
    train_preds = np.argmax(trainer.predict(train_dataset).predictions, axis=1)
    test_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=1)
    test_probs = torch.softmax(torch.tensor(test_output.predictions), dim=1).numpy()

    # Evaluate
    evaluate_model(
        model_key,
        y_train, train_preds,
        y_test, test_preds,
        classes, test_probs
    )

    # Store training time
    TRAINING_TIMES[model_key] = training_time

    # Save model
    model_dir = os.path.join(OUTPUT_DIR, f"{safe_name}_final")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    TRAINED_MODELS.append(model_key)

    print(f"\n✅ {model_key} training completed in {training_time/60:.2f} minutes")

    return model, tokenizer, trainer


# ==========================================================
# COMPREHENSIVE COMPARISON VISUALIZATION
# ==========================================================

def plot_comprehensive_comparison():
    """Create comprehensive comparison visualizations"""
    if not RESULTS:
        print("No results to visualize!")
        return

    df = pd.DataFrame(RESULTS).T
    df.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"))

    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(df.to_string())

    # Figure 1: Main Performance Metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bar chart comparison
    metrics = ["test_accuracy", "macro_f1", "weighted_f1"]
    colors = []
    for idx in df.index:
        # Try to find color from config
        color = None
        for config in MODELS_TO_COMPARE:
            if config["name"] in idx:
                color = config["color"]
                break
        colors.append(color if color else "#95a5a6")

    df[metrics].plot(kind="bar", ax=axes[0, 0], color=colors)
    axes[0, 0].set_title("Performance Metrics Comparison", fontsize=14)
    axes[0, 0].set_ylabel("Score", fontsize=12)
    axes[0, 0].set_ylim(0.8, 1.0)
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticklabels(df.index, rotation=45)

    # 2. Per-class F1 comparison (if we have at least 2 models)
    if len(TRAINED_MODELS) >= 2:
        classes = list(RESULTS[TRAINED_MODELS[0]]["per_class"].keys())

        axes[0, 1].set_title("Per-Class F1 Score Comparison", fontsize=14)
        x = np.arange(len(classes))
        width = 0.8 / len(TRAINED_MODELS)

        for i, model_name in enumerate(TRAINED_MODELS[:3]):  # Max 3 models for clarity
            f1_scores = [RESULTS[model_name]["per_class"][cls]["f1"] for cls in classes]
            offset = (i - (len(TRAINED_MODELS)-1)/2) * width
            # Get color from config
            color = None
            for config in MODELS_TO_COMPARE:
                if config["name"] in model_name:
                    color = config["color"]
                    break
            color = color if color else f"C{i}"
            axes[0, 1].bar(x + offset, f1_scores, width, label=model_name, color=color, alpha=0.7)

        axes[0, 1].set_xlabel("Disease Classes", fontsize=12)
        axes[0, 1].set_ylabel("F1 Score", fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=90, fontsize=8)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Training time comparison
    if TRAINING_TIMES:
        models = list(TRAINING_TIMES.keys())
        times = [TRAINING_TIMES[m] for m in models]
        colors = []
        for m in models:
            color = None
            for config in MODELS_TO_COMPARE:
                if config["name"] in m:
                    color = config["color"]
                    break
            colors.append(color if color else "#95a5a6")
        axes[1, 0].bar(models, times, color=colors)
        axes[1, 0].set_title("Training Time Comparison", fontsize=14)
        axes[1, 0].set_ylabel("Training Time (seconds)", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # Add values on bars
        for i, v in enumerate(times):
            axes[1, 0].text(i, v + 5, f"{v/60:.1f} min", ha="center")

    # 4. Summary text
    summary_text = "MODEL COMPARISON SUMMARY\n" + "="*30 + "\n\n"
    for model_name in TRAINED_MODELS:
        if model_name in RESULTS:
            r = RESULTS[model_name]
            summary_text += f"{model_name}:\n"
            summary_text += f"  • Test Accuracy: {r['test_accuracy']:.2%}\n"
            summary_text += f"  • Macro F1: {r['macro_f1']:.2%}\n"
            if r['macro_auc']:
                summary_text += f"  • Macro AUC: {r['macro_auc']:.2%}\n"
            if model_name in TRAINING_TIMES:
                summary_text += f"  • Training Time: {TRAINING_TIMES[model_name]/60:.1f} min\n"
            summary_text += "\n"

    # Find best model
    if TRAINED_MODELS:
        best_model = max(TRAINED_MODELS, key=lambda x: RESULTS.get(x, {}).get("test_accuracy", 0))
        if best_model in RESULTS:
            summary_text += f"🏆 BEST MODEL: {best_model}\n"
            summary_text += f"   Accuracy: {RESULTS[best_model]['test_accuracy']:.2%}"

    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    axes[1, 1].set_title("Summary", fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "comprehensive_comparison.png"), dpi=150)
    plt.close()
    print(f"✅ Comparison visualization saved to {FIG_DIR}/comprehensive_comparison.png")


def print_detailed_analysis():
    """Print detailed analysis with key insights"""
    if len(TRAINED_MODELS) < 1:
        print("\n⚠️ No models trained for analysis")
        return

    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    # Overall performance
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  {'Model':<30} {'Test Acc':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Time (min)':<12}")
    print(f"  {'-'*80}")

    for model_name in TRAINED_MODELS:
        if model_name in RESULTS:
            r = RESULTS[model_name]
            time_min = TRAINING_TIMES.get(model_name, 0) / 60
            print(f"  {model_name:<30} {r['test_accuracy']*100:>6.2f}%    {r['macro_f1']*100:>6.2f}%    {r['weighted_f1']*100:>6.2f}%    {time_min:>8.1f}")

    print(f"\n💡 KEY INSIGHTS:")
    print("""
    1. Model Comparison:
       • BERT-base: Standard BERT with 110M parameters
       • DistilBERT: Distilled version (40% smaller, 60% faster)
    
    2. Speed vs Accuracy Trade-off:
       • DistilBERT trains faster but may have slightly lower accuracy
       • BERT-base may achieve better accuracy but takes longer
    
    3. Practical Recommendations:
       • Use BERT-base for maximum accuracy
       • Use DistilBERT for production/deployment (speed + good accuracy)
    """)


# ==========================================================
# INFERENCE FUNCTIONS
# ==========================================================

def predict(model_key, text, model_dir=OUTPUT_DIR):
    """Make predictions using trained model"""
    safe_name = model_key.lower().replace("-", "_").replace(" ", "_")
    model_path = os.path.join(model_dir, f"{safe_name}_final")

    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    model.eval()
    inputs = tokenizer(text, truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Get top 5 predictions
    top_indices = np.argsort(probs)[-5:][::-1]
    results = []
    for idx in top_indices:
        disease = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        results.append((disease, confidence))

    return results


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("="*80)
    print("SYMPTOM-TO-DISEASE CLASSIFICATION: BERT vs DistilBERT")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and prepare data
    print("\n📥 Loading dataset...")
    df = load_data()
    X, y, label_encoder, classes = preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n📊 DATASET STATISTICS:")
    print(f"  • Total samples: {len(df)}")
    print(f"  • Train samples: {len(X_train)}")
    print(f"  • Test samples: {len(X_test)}")
    print(f"  • Classes: {len(classes)}")
    print(f"  • Max sequence length: {MAX_LEN}")
    print(f"  • Epochs: {EPOCHS}")

    # Train models
    for model_config in MODELS_TO_COMPARE:
        print(f"\n{'='*80}")
        print(f"Training: {model_config['name']}")
        print(f"{'='*80}")

        try:
            model, tokenizer, trainer = train_model(
                X_train, X_test, y_train, y_test, classes, model_config
            )
        except Exception as e:
            print(f"❌ Failed to train {model_config['name']}: {e}")
            continue

    if not TRAINED_MODELS:
        print("\n❌ No models could be trained. Exiting...")
        return

    # Visualize and analyze
    print("\n" + "="*80)
    print("VISUALIZING RESULTS")
    print("="*80)
    plot_comprehensive_comparison()
    print_detailed_analysis()

    # Test inference with sample
    sample = "I have severe headache, nausea, and sensitivity to light"
    print("\n" + "="*80)
    print("🔍 SAMPLE INFERENCE - TOP 5 PREDICTIONS")
    print("="*80)
    print(f"Input: {sample}\n")

    for model_name in TRAINED_MODELS[:3]:
        predictions = predict(model_name, sample)
        if predictions:
            print(f"{model_name}:")
            for i, (disease, conf) in enumerate(predictions, 1):
                bar = "█" * int(conf * 40)
                print(f"  {i}. {disease:25} {conf*100:5.1f}% {bar}")
            print()

    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print(f"📁 Results saved to '{OUTPUT_DIR}/' directory")
    print(f"📊 Models trained: {', '.join(TRAINED_MODELS)}")
    print("="*80)


if __name__ == "__main__":
    main()