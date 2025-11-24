import pandas as pd
import numpy as np
import warnings
import time

from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, \
    precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tabulate import tabulate
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set HD style for research papers with Times New Roman
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

plt.style.use('default')
sns.set_palette("husl")


def load_heart_disease_data():
    df = pd.read_csv('dataset/heart.csv')
    print("🔍 Dataset Overview:")
    print(f"Dataset Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    return df


def advanced_preprocessing(df):
    print("\n" + "=" * 50)
    print("🔄 ADVANCED PREPROCESSING")
    print("=" * 50)

    df_clean = df.copy()

    # Handle outliers using robust method
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers instead of removing
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])

    print("✅ Outlier handling completed")
    return df_clean


def enhanced_feature_engineering(df):
    print("\n" + "=" * 50)
    print("🔧 ENHANCED FEATURE ENGINEERING")
    print("=" * 50)

    df_engineered = df.copy()

    # Create meaningful features
    df_engineered['age_group'] = pd.cut(df_engineered['age'],
                                        bins=[0, 40, 50, 60, 100],
                                        labels=['Young', 'Middle', 'Senior', 'Elderly'])

    df_engineered['bp_category'] = pd.cut(df_engineered['trestbps'],
                                          bins=[0, 120, 140, 1000],
                                          labels=['Normal', 'Pre-High', 'High'])

    df_engineered['chol_category'] = pd.cut(df_engineered['chol'],
                                            bins=[0, 200, 240, 1000],
                                            labels=['Normal', 'Borderline', 'High'])

    # Enhanced interaction features
    df_engineered['age_bp_interaction'] = df_engineered['age'] * df_engineered['trestbps']
    df_engineered['hr_age_ratio'] = df_engineered['thalach'] / (df_engineered['age'] + 1)

    # New clinical features
    df_engineered['bp_hr_ratio'] = df_engineered['trestbps'] / (df_engineered['thalach'] + 1)
    df_engineered['chol_age_ratio'] = df_engineered['chol'] / (df_engineered['age'] + 1)
    df_engineered['risk_pressure_index'] = df_engineered['oldpeak'] * df_engineered['trestbps']

    # Polynomial features
    df_engineered['age_squared'] = df_engineered['age'] ** 2
    df_engineered['oldpeak_squared'] = df_engineered['oldpeak'] ** 2

    # Enhanced risk score
    df_engineered['clinical_risk'] = (
            (df_engineered['age'] > 50).astype(int) +
            (df_engineered['trestbps'] > 140).astype(int) +
            (df_engineered['chol'] > 240).astype(int) +
            (df_engineered['oldpeak'] > 1).astype(int) +
            df_engineered['exang']
    )

    print(f"✅ Created {len([col for col in df_engineered.columns if col not in df.columns])} new features")
    return df_engineered


def enhanced_feature_selection(X, y, n_features=10):
    """Enhanced feature selection using multiple methods"""
    print("\n🔍 Performing enhanced feature selection...")

    # Method 1: SelectKBest
    selector_kbest = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    X_kbest = selector_kbest.fit_transform(X, y)
    selected_features_kbest = X.columns[selector_kbest.get_support()]

    # Method 2: Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    top_rf_features = feature_importances.head(n_features)['feature'].tolist()

    # Combine both methods
    combined_features = list(set(selected_features_kbest.tolist() + top_rf_features))

    if len(combined_features) > n_features:
        # Use RF importance to select top n
        combined_features = \
            feature_importances[feature_importances['feature'].isin(combined_features)].head(n_features)[
                'feature'].tolist()

    print(f"✅ Selected {len(combined_features)} features using combined method")
    print(f"Selected features: {combined_features}")

    return X[combined_features], combined_features


def prepare_data(df):
    print("\n" + "=" * 50)
    print("📊 ENHANCED DATA PREPARATION")
    print("=" * 50)

    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # Enhanced feature selection
    X_selected, selected_features = enhanced_feature_selection(X, y, n_features=10)

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Testing set: {X_test_scaled.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features, df_encoded


def create_robust_cv_strategy():
    """Create more robust cross-validation strategy"""
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def create_regularized_mlp():
    """Create regularized MLP classifier to prevent overfitting"""
    return MLPClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        max_iter=1000,
        batch_size=32,
        alpha=0.01,  # Increased regularization
        learning_rate_init=0.001
    )


def create_ensemble_models():
    """Create ensemble models for improved performance"""
    base_models = [
        ('svm', SVC(C=1.0, gamma='scale', probability=True, random_state=42)),
        ('xgb', xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )),
        ('knn', KNeighborsClassifier(n_neighbors=15, weights='uniform', metric='manhattan'))
    ]

    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=base_models,
        voting='soft',
        n_jobs=-1
    )

    return {
        'Voting_Ensemble': voting_clf
    }


def calculate_feature_importance(model, model_name, X_test, y_test, selected_features):
    """Calculate feature importance using appropriate method for each model type"""

    if hasattr(model, 'feature_importances_'):
        # For tree-based models (XGBoost)
        importance = model.feature_importances_
        method = "Built-in Feature Importance"

    else:
        # For other models (SVM, MLP, KNN) - use permutation importance
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            importance = perm_importance.importances_mean
            method = "Permutation Importance"
        except:
            # Fallback: use coefficients if available
            if hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
                method = "Absolute Coefficients"
            else:
                # Last resort: uniform importance
                importance = np.ones(len(selected_features)) / len(selected_features)
                method = "Uniform (Fallback)"

    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importance,
        'Method': method
    }).sort_values('Importance', ascending=False)

    return feature_importance_df


def print_feature_importance_comprehensive(importance_data):
    """Print comprehensive feature importance analysis in terminal"""
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    for model_name, importance_df in importance_data.items():
        print(f"\n🎯 {model_name} - {importance_df['Method'].iloc[0]}:")
        print(tabulate(importance_df[['Feature', 'Importance']].round(4),
                       headers='keys', tablefmt='grid', showindex=False))


def create_roc_auc_visualization(best_models, X_test, y_test, filename='roc_auc_comparison_hd.png'):
    """Create and save ROC AUC visualization for all models in HD"""
    print(f"\n📊 Creating ROC AUC Visualization (HD)...")

    plt.figure(figsize=(12, 10))

    # Define colors for different models
    model_colors = {
        'SVM (RBF)': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'MLP': '#2ca02c',
        'KNN': '#d62728',
        'Voting_Ensemble': '#9467bd'
    }

    # Plot ROC curve for each model
    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, linewidth=3,
                     color=model_colors.get(model_name, 'black'),
                     label=f'{model_name} (AUC = {auc_score:.3f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Classifier (AUC = 0.500)')

    # Customize the plot for research paper
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves\nModel Comparison',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # Add performance annotations
    plt.text(0.6, 0.05, 'Better Models → Top Left',
             fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    # Save the plot in HD
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD ROC AUC visualization saved as '{filename}'")


def create_comprehensive_model_comparison(best_models, X_test, y_test, results, selected_features,
                                          filename='comprehensive_model_comparison_hd.png'):
    """Create a comprehensive visualization with all models and metrics in HD"""
    print(f"\n📊 Creating Comprehensive Model Comparison Dashboard (HD)...")

    # Create a 2x2 subplot for comprehensive comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Model Evaluation Dashboard', fontsize=18, fontweight='bold')

    # Colors for different models
    model_colors = {
        'SVM (RBF)': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'MLP': '#2ca02c',
        'KNN': '#d62728',
        'Voting_Ensemble': '#9467bd'
    }

    # 1. ROC Curves (Top Left)
    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            axes[0, 0].plot(fpr, tpr, linewidth=3, color=model_colors[model_name],
                            label=f'{model_name} (AUC = {auc_score:.3f})')

    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Classifier')
    axes[0, 0].set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    axes[0, 0].set_title('ROC Curves', fontweight='bold', fontsize=14)
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curves (Top Right)
    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            axes[0, 1].plot(recall, precision, linewidth=3, color=model_colors[model_name],
                            label=f'{model_name} (AP = {avg_precision:.3f})')

    axes[0, 1].set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontweight='bold', fontsize=12)
    axes[0, 1].set_title('Precision-Recall Curves', fontweight='bold', fontsize=14)
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Performance Metrics Bar Chart (Bottom Left)
    model_names = [result['Model'] for result in results]
    test_accuracies = [float(result['Test Acc']) for result in results]
    auc_scores = [float(result['AUC Score']) if result['AUC Score'] != 'N/A' else 0 for result in results]
    cv_scores = [float(result['CV Score']) for result in results]

    x = np.arange(len(model_names))
    width = 0.25

    # Use model colors for bars
    colors = [model_colors.get(name, 'gray') for name in model_names]

    bars1 = axes[1, 0].bar(x - width, test_accuracies, width, label='Test Accuracy', alpha=0.8,
                           color=[model_colors.get(name, 'skyblue') for name in model_names])
    bars2 = axes[1, 0].bar(x, auc_scores, width, label='AUC Score', alpha=0.8,
                           color=[model_colors.get(name, 'lightcoral') for name in model_names])
    bars3 = axes[1, 0].bar(x + width, cv_scores, width, label='CV Score', alpha=0.8,
                           color=[model_colors.get(name, 'lightgreen') for name in model_names])

    axes[1, 0].set_xlabel('Models', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('Scores', fontweight='bold', fontsize=12)
    axes[1, 0].set_title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels
    for i, (acc, auc, cv) in enumerate(zip(test_accuracies, auc_scores, cv_scores)):
        axes[1, 0].text(i - width, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        axes[1, 0].text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
        axes[1, 0].text(i + width, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Calibration Curves (Bottom Right)
    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
            axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2,
                            color=model_colors[model_name], label=f'{model_name}')

    axes[1, 1].plot([0, 1], [0, 1], 'k:', linewidth=2, label='Perfectly calibrated')
    axes[1, 1].set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=12)
    axes[1, 1].set_ylabel('Fraction of Positives', fontweight='bold', fontsize=12)
    axes[1, 1].set_title('Calibration Curves', fontweight='bold', fontsize=14)
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD comprehensive model comparison saved as '{filename}'")


def create_learning_curves_visualization(best_models, X_train, y_train, filename='learning_curves_comparison_hd.png'):
    """Create learning curves visualization for all models in HD"""
    print(f"\n📈 Creating Learning Curves Visualization (HD)...")

    plt.figure(figsize=(14, 10))

    # Define colors for different models
    model_colors = {
        'SVM (RBF)': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'MLP': '#2ca02c',
        'KNN': '#d62728',
        'Voting_Ensemble': '#9467bd'
    }

    for model_name, model in best_models.items():
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        color = model_colors.get(model_name, 'black')

        plt.plot(train_sizes, train_scores_mean, 'o-', linewidth=2.5, color=color,
                 label=f'{model_name} - Training')
        plt.plot(train_sizes, test_scores_mean, 'o--', linewidth=2.5, color=color,
                 label=f'{model_name} - Cross-validation')

        # Fill between standard deviation areas
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color=color)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color=color)

    plt.xlabel('Training Examples', fontweight='bold', fontsize=12)
    plt.ylabel('Accuracy Score', fontweight='bold', fontsize=12)
    plt.title('Learning Curves - All Models', fontweight='bold', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD learning curves comparison saved as '{filename}'")


def create_clinical_distributions_consolidated(df, filename='clinical_distributions_consolidated_hd.png'):
    """Create consolidated clinical distributions visualization in one graph"""
    print(f"\n📊 Creating Consolidated Clinical Distributions Visualization (HD)...")

    plt.figure(figsize=(16, 10))

    # Define clinical features and their labels
    clinical_features = ['restecg', 'cp', 'slope', 'exang']
    feature_labels = {
        'restecg': 'Resting ECG',
        'cp': 'Chest Pain Type',
        'slope': 'Slope of ST Segment',
        'exang': 'Exercise Induced Angina'
    }

    # Define colors for different clinical features
    feature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calculate disease rates for each feature category
    disease_rates_data = []

    for feature, color in zip(clinical_features, feature_colors):
        if feature in df.columns:
            # Calculate disease rate for each category
            feature_dist = df.groupby(feature)['target'].agg(['count', 'mean']).reset_index()
            feature_dist['disease_rate'] = feature_dist['mean'] * 100
            feature_dist['feature_type'] = feature_labels[feature]
            feature_dist['color'] = color

            disease_rates_data.append(feature_dist)

    # Combine all data
    all_disease_rates = pd.concat(disease_rates_data, ignore_index=True)

    # Create the plot
    ax = plt.subplot(111)

    # Plot bars for each feature category
    bar_width = 0.6
    y_pos = np.arange(len(all_disease_rates))

    bars = ax.barh(y_pos, all_disease_rates['disease_rate'],
                   height=bar_width, color=all_disease_rates['color'], alpha=0.8, edgecolor='black')

    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['feature_type']} - Category {row[feature]}"
                        for _, row in all_disease_rates.iterrows()], fontsize=10)
    ax.set_xlabel('Heart Disease Rate (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Clinical Feature Categories', fontweight='bold', fontsize=12)
    ax.set_title('Clinical Feature Distributions - Heart Disease Rates by Category',
                 fontweight='bold', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, rate, count) in enumerate(zip(bars, all_disease_rates['disease_rate'], all_disease_rates['count'])):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1f}% (n={count})', va='center', fontsize=9, fontweight='bold')

    # Create custom legend for clinical features
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, edgecolor='black',
                                     label=feature_labels[feature])
                       for feature, color in zip(clinical_features, feature_colors)]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD consolidated clinical distributions saved as '{filename}'")


def create_feature_importance_comparison(best_models, X_test, y_test, selected_features,
                                         filename='feature_importance_comparison_hd.png'):
    """Create comprehensive feature importance comparison across all models in one graph"""
    print(f"\n📊 Creating Feature Importance Comparison (HD)...")

    # Calculate feature importance for all models
    importance_data = {}
    for model_name, model in best_models.items():
        importance_data[model_name] = calculate_feature_importance(
            model, model_name, X_test, y_test, selected_features
        )

    # Create visualization
    plt.figure(figsize=(16, 12))

    # Define colors for models
    model_colors = {
        'SVM (RBF)': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'MLP': '#2ca02c',
        'KNN': '#d62728',
        'Voting_Ensemble': '#9467bd'
    }

    # Get all unique features across models
    all_features = set()
    for importance_df in importance_data.values():
        all_features.update(importance_df['Feature'].tolist())

    # Create a matrix for heatmap
    feature_matrix = pd.DataFrame(index=list(all_features))

    for model_name, importance_df in importance_data.items():
        # Normalize importance scores to 0-1 for better comparison
        importance_df_normalized = importance_df.copy()
        importance_df_normalized['Importance'] = importance_df_normalized['Importance'] / importance_df_normalized[
            'Importance'].max()

        # Create mapping for features
        feature_importance_map = dict(zip(importance_df_normalized['Feature'], importance_df_normalized['Importance']))
        feature_matrix[model_name] = feature_matrix.index.map(lambda x: feature_importance_map.get(x, 0))

    # Sort features by average importance
    feature_matrix['Average'] = feature_matrix.mean(axis=1)
    feature_matrix = feature_matrix.sort_values('Average', ascending=True)
    feature_matrix = feature_matrix.drop('Average', axis=1)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(feature_matrix.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Customize axes
    ax.set_xticks(np.arange(len(feature_matrix)))
    ax.set_yticks(np.arange(len(feature_matrix.columns)))
    ax.set_xticklabels(feature_matrix.index, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(feature_matrix.columns, fontsize=12, fontweight='bold')

    # Add value annotations
    for i in range(len(feature_matrix.columns)):
        for j in range(len(feature_matrix)):
            text = ax.text(j, i, f'{feature_matrix.iloc[j, i]:.2f}',
                           ha="center", va="center", color="black" if feature_matrix.iloc[j, i] < 0.6 else "white",
                           fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Feature Importance', fontweight='bold', fontsize=12)

    plt.title('Comprehensive Feature Importance Comparison Across All Models',
              fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD feature importance comparison saved as '{filename}'")
    return importance_data


def create_prediction_confidence_distribution(best_models, X_test, y_test,
                                              filename='prediction_confidence_distribution_hd.png'):
    """Create distribution of prediction confidence scores for all models in one graph"""
    print(f"\n📊 Creating Prediction Confidence Distribution (HD)...")

    plt.figure(figsize=(14, 10))

    # Define colors for models
    model_colors = {
        'SVM (RBF)': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'MLP': '#2ca02c',
        'KNN': '#d62728',
        'Voting_Ensemble': '#9467bd'
    }

    # Create subplots for correct and incorrect predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Plot correct vs incorrect predictions
            correct_predictions = (model.predict(X_test) == y_test)
            correct_confidences = y_pred_proba[correct_predictions]
            incorrect_confidences = y_pred_proba[~correct_predictions]

            # Plot density distributions
            if len(correct_confidences) > 0:
                ax1.hist(correct_confidences, bins=20, alpha=0.6, color=model_colors[model_name],
                         label=f'{model_name}', density=True, histtype='stepfilled')

            if len(incorrect_confidences) > 0:
                ax2.hist(incorrect_confidences, bins=20, alpha=0.6, color=model_colors[model_name],
                         label=f'{model_name}', density=True, histtype='stepfilled')

    # Customize correct predictions subplot
    ax1.set_xlabel('Prediction Confidence', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax1.set_title('Confidence Distribution - Correct Predictions', fontweight='bold', fontsize=14)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Customize incorrect predictions subplot
    ax2.set_xlabel('Prediction Confidence', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax2.set_title('Confidence Distribution - Incorrect Predictions', fontweight='bold', fontsize=14)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD prediction confidence distribution saved as '{filename}'")


def create_model_performance_parallel(best_models, X_test, y_test, filename='model_performance_parallel_hd.png'):
    """Create parallel coordinates plot for model performance metrics"""
    print(f"\n📊 Creating Model Performance Parallel Coordinates (HD)...")

    # Calculate metrics for each model
    metrics_data = []

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0

        metrics_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        })

    # Create parallel coordinates plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define colors for models
    model_colors = {
        'SVM (RBF)': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'MLP': '#2ca02c',
        'KNN': '#d62728',
        'Voting_Ensemble': '#9467bd'
    }

    # Prepare data for parallel coordinates
    metrics_df = pd.DataFrame(metrics_data)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    # Normalize metrics to 0-1 scale for better visualization
    normalized_df = metrics_df.copy()
    for metric in metrics:
        normalized_df[metric] = (normalized_df[metric] - normalized_df[metric].min()) / (
                    normalized_df[metric].max() - normalized_df[metric].min())

    # Plot parallel coordinates
    for idx, row in normalized_df.iterrows():
        model_name = row['Model']
        values = [row[metric] for metric in metrics]
        ax.plot(metrics, values, 'o-', linewidth=3, markersize=8,
                color=model_colors[model_name], label=model_name, alpha=0.8)

    # Customize the plot
    ax.set_ylabel('Normalized Performance Score', fontweight='bold', fontsize=12)
    ax.set_xlabel('Performance Metrics', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison - Parallel Coordinates',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Add value annotations
    for idx, row in metrics_df.iterrows():
        model_name = row['Model']
        for i, metric in enumerate(metrics):
            ax.text(i, normalized_df.loc[idx, metric] + 0.02, f'{row[metric]:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=model_colors[model_name], alpha=0.7))

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ HD model performance parallel coordinates saved as '{filename}'")


def print_model_performance_parallel(best_models, X_test, y_test):
    """Print parallel coordinates metrics in terminal"""
    print("\n" + "=" * 80)
    print("📊 MODEL PERFORMANCE PARALLEL COORDINATES METRICS")
    print("=" * 80)

    parallel_data = []

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0

        parallel_data.append([
            model_name,
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{auc_roc:.4f}"
        ])

    print(tabulate(parallel_data,
                   headers=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                   tablefmt='grid'))


def print_prediction_confidence_analysis(best_models, X_test, y_test):
    """Print prediction confidence analysis in terminal"""
    print("\n" + "=" * 80)
    print("📊 PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 80)

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            correct_predictions = (y_pred == y_test)
            correct_confidences = y_pred_proba[correct_predictions]
            incorrect_confidences = y_pred_proba[~correct_predictions]

            print(f"\n🔍 {model_name} Confidence Analysis:")
            print(f"   Correct Predictions: {len(correct_confidences)}")
            print(f"   Incorrect Predictions: {len(incorrect_confidences)}")

            if len(correct_confidences) > 0:
                print(f"   Avg Confidence (Correct): {np.mean(correct_confidences):.4f}")
                print(f"   Std Confidence (Correct): {np.std(correct_confidences):.4f}")

            if len(incorrect_confidences) > 0:
                print(f"   Avg Confidence (Incorrect): {np.mean(incorrect_confidences):.4f}")
                print(f"   Std Confidence (Incorrect): {np.std(incorrect_confidences):.4f}")

            # Calculate confidence threshold analysis
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            print(f"\n   Confidence Threshold Analysis:")
            for threshold in thresholds:
                high_confidence = y_pred_proba >= threshold
                high_confidence_correct = np.sum((y_pred == y_test) & high_confidence)
                high_confidence_total = np.sum(high_confidence)

                if high_confidence_total > 0:
                    accuracy_at_threshold = high_confidence_correct / high_confidence_total
                    print(f"   Threshold {threshold}: {accuracy_at_threshold:.4f} accuracy "
                          f"({high_confidence_correct}/{high_confidence_total} predictions)")


def print_correlation_matrix(df, selected_features, target_col='target'):
    """Print correlation matrix in terminal"""
    print("\n" + "=" * 80)
    print("📊 CORRELATION MATRIX ANALYSIS")
    print("=" * 80)

    # Include target in correlation analysis
    features_to_plot = selected_features + [target_col]
    corr_matrix = df[features_to_plot].corr()

    print("\nCorrelation Matrix (Features vs Target):")
    print("=" * 50)

    # Print correlation with target
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    target_corr_df = pd.DataFrame({
        'Feature': target_corr.index,
        'Correlation with Target': target_corr.values
    }).round(4)

    print(tabulate(target_corr_df, headers='keys', tablefmt='grid', showindex=False))

    print(f"\nTop 5 Most Correlated Features with Target:")
    print("=" * 50)
    top_corr = target_corr.abs().sort_values(ascending=False).head(6)  # Include target itself
    top_corr = top_corr[top_corr.index != target_col]  # Remove target
    top_corr_df = pd.DataFrame({
        'Feature': top_corr.index,
        'Absolute Correlation': top_corr.values
    }).round(4)

    print(tabulate(top_corr_df, headers='keys', tablefmt='grid', showindex=False))


def print_feature_distributions(df, selected_features, target_col='target'):
    """Print feature distribution statistics in terminal"""
    print("\n" + "=" * 80)
    print("📈 FEATURE DISTRIBUTION STATISTICS BY TARGET CLASS")
    print("=" * 80)

    # Select top features for clear visualization
    top_features = selected_features[:8] if len(selected_features) >= 8 else selected_features

    for feature in top_features:
        print(f"\n📊 {feature} Distribution:")
        stats_df = df.groupby(target_col)[feature].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
        stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'Count']
        print(tabulate(stats_df, headers='keys', tablefmt='grid'))


def print_learning_curves(models, X_train, y_train):
    """Print learning curve summary in terminal"""
    print("\n" + "=" * 80)
    print("📈 LEARNING CURVES ANALYSIS SUMMARY")
    print("=" * 80)

    learning_data = []

    for model_name, model in models.items():
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),  # Reduced points for faster computation
            scoring='accuracy', random_state=42
        )

        train_scores_final = np.mean(train_scores[-1])  # Final training score
        test_scores_final = np.mean(test_scores[-1])  # Final test score
        gap = train_scores_final - test_scores_final

        learning_data.append([
            model_name,
            f"{train_scores_final:.4f}",
            f"{test_scores_final:.4f}",
            f"{gap:.4f}",
            "🟥 High" if gap > 0.05 else "🟨 Medium" if gap > 0.02 else "🟩 Low"
        ])

    print(tabulate(learning_data,
                   headers=['Model', 'Final Train Acc', 'Final CV Acc', 'Gap', 'Overfitting Risk'],
                   tablefmt='grid'))


def print_confusion_matrices(best_models, X_test, y_test):
    """Print confusion matrices in terminal"""
    print("\n" + "=" * 80)
    print("🎯 CONFUSION MATRICES ANALYSIS")
    print("=" * 80)

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score_val = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        print(f"\n🔍 {model_name} Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Sensitivity: {sensitivity:.4f}")
        print(f"   Specificity: {specificity:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   F1-Score: {f1_score_val:.4f}")

        print(f"\nConfusion Matrix:")
        cm_df = pd.DataFrame(cm,
                             index=['True No Disease', 'True Disease'],
                             columns=['Pred No Disease', 'Pred Disease'])
        print(tabulate(cm_df, headers='keys', tablefmt='grid'))


def print_roc_analysis(best_models, X_test, y_test):
    """Print ROC-AUC analysis in terminal"""
    print("\n" + "=" * 80)
    print("📈 ROC-AUC ANALYSIS")
    print("=" * 80)

    auc_data = []

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)

            # Calculate ROC curve points
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

            # Find optimal threshold (Youden's J statistic)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]

            auc_data.append([
                model_name,
                f"{auc_score:.4f}",
                f"{optimal_threshold:.4f}",
                f"{tpr[optimal_idx]:.4f}",  # Sensitivity at optimal threshold
                f"{1 - fpr[optimal_idx]:.4f}"  # Specificity at optimal threshold
            ])
        else:
            auc_data.append([model_name, "N/A", "N/A", "N/A", "N/A"])

    print(tabulate(auc_data,
                   headers=['Model', 'AUC Score', 'Optimal Threshold', 'Sensitivity', 'Specificity'],
                   tablefmt='grid'))


def print_calibration_analysis(best_models, X_test, y_test):
    """Print calibration analysis in terminal"""
    print("\n" + "=" * 80)
    print("⚖️ MODEL CALIBRATION ANALYSIS")
    print("=" * 80)

    calibration_data = []

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

            # Calculate calibration error (mean absolute error)
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

            calibration_data.append([
                model_name,
                f"{calibration_error:.4f}",
                "Well Calibrated" if calibration_error < 0.1 else "Moderately Calibrated" if calibration_error < 0.2 else "Poorly Calibrated"
            ])
        else:
            calibration_data.append([model_name, "N/A", "N/A"])

    print(tabulate(calibration_data,
                   headers=['Model', 'Calibration Error', 'Calibration Quality'],
                   tablefmt='grid'))


def print_prediction_probabilities(best_models, X_test, y_test):
    """Print prediction probability statistics in terminal"""
    print("\n" + "=" * 80)
    print("📊 PREDICTION PROBABILITY STATISTICS")
    print("=" * 80)

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            stats_data = []

            for target_val in [0, 1]:
                mask = (y_test == target_val)
                probs = y_pred_proba[mask]
                if len(probs) > 0:
                    stats_data.append([
                        'No Disease' if target_val == 0 else 'Disease',
                        f"{np.mean(probs):.3f}",
                        f"{np.std(probs):.3f}",
                        f"{np.min(probs):.3f}",
                        f"{np.median(probs):.3f}",
                        f"{np.max(probs):.3f}",
                        f"{len(probs)}"
                    ])

            print(f"\n📈 {model_name} Probability Distribution:")
            print(tabulate(stats_data,
                           headers=['True Class', 'Mean', 'Std', 'Min', 'Median', 'Max', 'Count'],
                           tablefmt='grid'))


def print_classification_reports(best_models, X_test, y_test):
    """Print classification reports for all models"""
    print("\n" + "=" * 80)
    print("📋 DETAILED CLASSIFICATION REPORTS - ALL MODELS")
    print("=" * 80)

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        print(f"\n🎯 {model_name} Classification Report:")
        print("=" * 50)

        # Print main metrics
        main_metrics = report_df.loc[['0', '1', 'accuracy'], :].round(4)
        print(tabulate(main_metrics, headers='keys', tablefmt='grid'))

        # Print weighted averages
        weighted_avg = report_df.loc[['weighted avg'], :].round(4)
        print(f"\nWeighted Averages:")
        print(tabulate(weighted_avg, headers='keys', tablefmt='grid'))


def print_clinical_distributions(df):
    """Print clinical distributions in terminal"""
    print("\n" + "=" * 80)
    print("🏥 CLINICAL FEATURE DISTRIBUTIONS")
    print("=" * 80)

    clinical_features = ['restecg', 'cp', 'slope', 'exang', 'thal']
    feature_names = {
        'restecg': 'Resting ECG',
        'cp': 'Chest Pain Type',
        'slope': 'Slope of Peak Exercise ST Segment',
        'exang': 'Exercise Induced Angina',
        'thal': 'Thalassemia'
    }

    for feature in clinical_features:
        if feature in df.columns:
            print(f"\n📊 {feature_names.get(feature, feature)} Distribution:")
            distribution = pd.crosstab(df[feature], df['target'])
            distribution.columns = ['No Disease', 'Disease']
            distribution['Total'] = distribution.sum(axis=1)
            distribution['Disease Rate (%)'] = (distribution['Disease'] / distribution['Total'] * 100).round(2)
            print(tabulate(distribution, headers='keys', tablefmt='grid'))


def create_comprehensive_visualizations(best_models, X_test, y_test, X_train, y_train, selected_features, results, df):
    """Create comprehensive visualizations including ROC AUC in HD"""
    print(f"\n📊 Creating All Comprehensive Visualizations (HD)...")

    # 1. Main comprehensive comparison (4-in-1)
    create_comprehensive_model_comparison(best_models, X_test, y_test, results, selected_features)

    # 2. Learning curves
    create_learning_curves_visualization(best_models, X_train, y_train)

    # 3. ROC AUC
    create_roc_auc_visualization(best_models, X_test, y_test)

    # 4. Clinical distributions (CONSOLIDATED - 1 graph)
    create_clinical_distributions_consolidated(df)

    # 5. NEW: Model Performance Parallel Coordinates
    create_model_performance_parallel(best_models, X_test, y_test)

    # 6. NEW: Prediction Confidence Distribution (CONSOLIDATED - 1 graph)
    create_prediction_confidence_distribution(best_models, X_test, y_test)

    # 7. NEW: Feature Importance Comparison (CONSOLIDATED - 1 graph)
    importance_data = create_feature_importance_comparison(best_models, X_test, y_test, selected_features)

    # Print all visualizations in terminal
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TERMINAL OUTPUT FOR ALL VISUALIZATIONS")
    print("=" * 80)

    print_model_performance_parallel(best_models, X_test, y_test)
    print_prediction_confidence_analysis(best_models, X_test, y_test)

    return importance_data


def enhanced_train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_features, df_encoded):
    print("\n" + "=" * 80)
    print("🤖 ENHANCED MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # Create initial analyses
    print("\n📊 Creating initial analyses...")
    print_correlation_matrix(df_encoded, selected_features)
    print_feature_distributions(df_encoded, selected_features)
    print_clinical_distributions(df_encoded)

    cv_strategy = create_robust_cv_strategy()

    # Define models with better defaults to prevent overfitting
    models = {
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'XGBoost': xgb.XGBClassifier(
            random_state=42,
            max_depth=3,  # Reduced to prevent overfitting
            subsample=0.8
        ),
        'MLP': create_regularized_mlp(),
        'KNN': KNeighborsClassifier(n_neighbors=15),  # Start with higher k to reduce overfitting
        **create_ensemble_models()  # Add ensemble methods
    }

    # Updated parameter grids with focus on reducing overfitting
    param_grids = {
        'SVM (RBF)': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (30, 30), (50, 25, 10)],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01]
        },
        'KNN': {
            'n_neighbors': [15, 20, 25, 30],  # Higher values to reduce overfitting
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        },
        'Voting_Ensemble': {
            'voting': ['soft', 'hard']
        }
    }

    results = []
    best_models = {}
    training_times = {}

    for model_name, model in models.items():
        print(f"\n📈 Training {model_name}...")

        start_time = time.time()

        # Skip grid search for ensemble if no parameters to tune
        if model_name == 'Voting_Ensemble' and not param_grids[model_name]:
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = "Default parameters"
        else:
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=cv_strategy,
                scoring='accuracy', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = str(grid_search.best_params_)

        training_time = time.time() - start_time
        training_times[model_name] = training_time

        best_models[model_name] = best_model

        y_pred = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy)
        cv_mean = cv_scores.mean()

        accuracy_gap = train_accuracy - test_accuracy
        overfitting_level = 'High' if accuracy_gap > 0.05 else 'Medium' if accuracy_gap > 0.02 else 'Low'

        results.append({
            'Model': model_name,
            'Best Params': best_params,
            'Train Acc': f"{train_accuracy:.4f}",
            'Test Acc': f"{test_accuracy:.4f}",
            'CV Score': f"{cv_mean:.4f}",
            'AUC Score': f"{auc_score:.4f}" if auc_score else 'N/A',
            'Acc Gap': f"{accuracy_gap:.4f}",
            'Overfitting': overfitting_level,
            'Training Time': f"{training_time:.2f}s"
        })

    # Create comprehensive analyses
    print("\n📈 Creating comprehensive analyses...")
    print_learning_curves(models, X_train, y_train)
    print_confusion_matrices(best_models, X_test, y_test)
    print_roc_analysis(best_models, X_test, y_test)
    print_calibration_analysis(best_models, X_test, y_test)
    print_prediction_probabilities(best_models, X_test, y_test)
    print_classification_reports(best_models, X_test, y_test)

    # Create comprehensive visualizations
    importance_data = create_comprehensive_visualizations(best_models, X_test, y_test, X_train, y_train,
                                                          selected_features, results, df_encoded)

    # Print comprehensive feature importance analysis
    print_feature_importance_comprehensive(importance_data)

    # Display results
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

    return best_models, results, importance_data


def main():
    print("🚀 HEART DISEASE CLASSIFICATION PROJECT - ENHANCED VERSION")
    print("=" * 80)
    print("Selected Models: SVM (RBF), XGBoost, MLP, KNN, Voting Ensemble")
    print("Key Improvements:")
    print("  • HD Visualizations with Times New Roman font")
    print("  • Enhanced feature engineering with clinical features")
    print("  • Improved feature selection using multiple methods")
    print("  • Regularized models to prevent overfitting")
    print("  • Higher K values for KNN (15-30)")
    print("  • Ensemble methods for better generalization")
    print("  • Comprehensive terminal output for research analysis")
    print("=" * 80)

    # 1. Load data
    df = load_heart_disease_data()

    # 2. Advanced preprocessing
    df_clean = advanced_preprocessing(df)

    # 3. Enhanced feature engineering
    df_engineered = enhanced_feature_engineering(df_clean)

    # 4. Prepare data with enhanced feature selection
    X_train, X_test, y_train, y_test, scaler, selected_features, df_encoded = prepare_data(df_engineered)

    # 5. Train and evaluate models with enhanced methods
    best_models, results, importance_data = enhanced_train_and_evaluate_models(
        X_train, X_test, y_train, y_test, selected_features, df_encoded
    )

    print("\n" + "🎯 PROJECT COMPLETED SUCCESSFULLY!")
    print("\n📊 ALL ANALYSES DISPLAYED IN TERMINAL ABOVE:")
    print("   ✓ Correlation Matrix Analysis")
    print("   ✓ Feature Distribution Statistics")
    print("   ✓ Clinical Feature Distributions")
    print("   ✓ Learning Curves Analysis")
    print("   ✓ Confusion Matrices Analysis")
    print("   ✓ ROC-AUC Analysis")
    print("   ✓ Calibration Analysis")
    print("   ✓ Prediction Probability Statistics")
    print("   ✓ Detailed Classification Reports")
    print("   ✓ Feature Importance Analysis")
    print("   ✓ Comprehensive Model Comparison")
    print("   ✓ Model Performance Parallel Coordinates Metrics")
    print("   ✓ Prediction Confidence Analysis")

    print("\n📈 HD VISUALIZATIONS SAVED:")
    print("   1. comprehensive_model_comparison_hd.png - Main dashboard (4 plots)")
    print("   2. learning_curves_comparison_hd.png - Learning curves")
    print("   3. roc_auc_comparison_hd.png - ROC curves")
    print("   4. clinical_distributions_consolidated_hd.png - Clinical distributions (CONSOLIDATED)")
    print("   5. model_performance_parallel_hd.png - Performance parallel coordinates")
    print("   6. prediction_confidence_distribution_hd.png - Confidence distributions (CONSOLIDATED)")
    print("   7. feature_importance_comparison_hd.png - Feature importance comparison (CONSOLIDATED)")

    print("\n🔍 FEATURE IMPORTANCE METHODS USED:")
    for model_name, importance_df in importance_data.items():
        print(f"   - {model_name}: {importance_df['Method'].iloc[0]}")

    print("\n✅ Key improvements implemented:")
    print("   • HD visualizations with Times New Roman font (600 DPI)")
    print("   • 3 CONSOLIDATED visualizations (clinical, confidence, feature importance)")
    print("   • Parallel coordinates plot instead of radar chart")
    print("   • KNN overfitting reduced with higher n_neighbors (15-30)")
    print("   • Enhanced feature engineering with clinical ratios")
    print("   • Improved feature selection using combined methods")
    print("   • Added ensemble methods for better generalization")
    print("   • Increased regularization across all models")
    print("   • Comprehensive terminal output for research analysis")


if __name__ == "__main__":
    main()