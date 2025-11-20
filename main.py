import pandas as pd
import numpy as np
import warnings

from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, \
    precision_recall_curve, average_precision_score
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

# Set style for better plots
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
        feature_importances[feature_importances['feature'].isin(combined_features)].head(n_features)['feature'].tolist()

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


def create_feature_importance_comprehensive(best_models, X_test, y_test, selected_features):
    """Create comprehensive feature importance comparison for all models"""
    print(f"\n🔍 Creating Comprehensive Feature Importance Analysis...")

    # Calculate feature importance for all models
    importance_data = {}
    for model_name, model in best_models.items():
        importance_data[model_name] = calculate_feature_importance(
            model, model_name, X_test, y_test, selected_features
        )

    # Create visualization
    n_models = len(importance_data)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, (model_name, importance_df) in enumerate(importance_data.items()):
        if idx >= len(axes):
            break

        # Plot horizontal bar chart
        ax = axes[idx]
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))

        ax.set_title(f'{model_name}\n({importance_df["Method"].iloc[0]})',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('Importance Score')

        # Add value labels on bars
        for bar, importance in zip(bars, importance_df['Importance']):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{importance:.3f}', ha='left', va='center', fontsize=9)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('comprehensive_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Comprehensive feature importance saved as 'comprehensive_feature_importance.png'")

    return importance_data


def print_feature_importance_comprehensive(importance_data):
    """Print comprehensive feature importance analysis in terminal"""
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    for model_name, importance_df in importance_data.items():
        print(f"\n🎯 {model_name} - {importance_df['Method'].iloc[0]}:")
        print(tabulate(importance_df[['Feature', 'Importance']].round(4),
                       headers='keys', tablefmt='grid', showindex=False))


def create_roc_auc_visualization(best_models, X_test, y_test, filename='roc_auc_comparison.png'):
    """Create and save ROC AUC visualization for all models"""
    print(f"\n📊 Creating ROC AUC Visualization...")

    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each model
    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, linewidth=2.5,
                     label=f'{model_name} (AUC = {auc_score:.3f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier (AUC = 0.500)')

    # Customize the plot
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add performance annotations
    plt.text(0.6, 0.05, '🔍 Better Models → Top Left',
             fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC AUC visualization saved as '{filename}'")


def create_probability_distributions_combined(best_models, X_test, y_test):
    """Create combined probability distribution plot for all models in one graph"""
    print(f"\n📊 Creating Combined Probability Distributions Visualization...")

    plt.figure(figsize=(14, 10))

    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Create subplots: left for No Disease, right for Disease
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for idx, (model_name, model) in enumerate(best_models.items()):
        if not hasattr(model, 'predict_proba'):
            continue

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        color = colors[idx % len(colors)]

        # Plot for No Disease class (y_test == 0)
        mask_no_disease = (y_test == 0)
        probs_no_disease = y_pred_proba[mask_no_disease]

        # Plot for Disease class (y_test == 1)
        mask_disease = (y_test == 1)
        probs_disease = y_pred_proba[mask_disease]

        # Plot histograms with KDE
        ax1.hist(probs_no_disease, bins=20, alpha=0.6, color=color,
                 label=f'{model_name}', density=True, histtype='stepfilled')
        ax2.hist(probs_disease, bins=20, alpha=0.6, color=color,
                 label=f'{model_name}', density=True, histtype='stepfilled')

    # Customize No Disease subplot
    ax1.set_xlabel('Predicted Probability', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Probability Distribution - No Disease Cases', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Customize Disease subplot
    ax2.set_xlabel('Predicted Probability', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Probability Distribution - Disease Cases', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('probability_distributions_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Combined probability distributions saved as 'probability_distributions_combined.png'")


def create_comprehensive_model_comparison(best_models, X_test, y_test, results, selected_features):
    """Create a comprehensive visualization with all models and metrics"""
    print(f"\n📊 Creating Comprehensive Model Comparison Dashboard...")

    # Create a 2x2 subplot for comprehensive comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Evaluation Dashboard', fontsize=16, fontweight='bold')

    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. ROC Curves (Top Left)
    for idx, (model_name, model) in enumerate(best_models.items()):
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            axes[0, 0].plot(fpr, tpr, linewidth=2.5, color=colors[idx],
                            label=f'{model_name} (AUC = {auc_score:.3f})')

    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    axes[0, 0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0, 0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[0, 0].set_title('ROC Curves', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curves (Top Right)
    for idx, (model_name, model) in enumerate(best_models.items()):
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            axes[0, 1].plot(recall, precision, linewidth=2.5, color=colors[idx],
                            label=f'{model_name} (AP = {avg_precision:.3f})')

    axes[0, 1].set_xlabel('Recall (Sensitivity)', fontweight='bold')
    axes[0, 1].set_ylabel('Precision', fontweight='bold')
    axes[0, 1].set_title('Precision-Recall Curves', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Performance Metrics Bar Chart (Bottom Left)
    model_names = [result['Model'] for result in results]
    test_accuracies = [float(result['Test Acc']) for result in results]
    auc_scores = [float(result['AUC Score']) if result['AUC Score'] != 'N/A' else 0 for result in results]
    cv_scores = [float(result['CV Score']) for result in results]

    x = np.arange(len(model_names))
    width = 0.25

    axes[1, 0].bar(x - width, test_accuracies, width, label='Test Accuracy', alpha=0.8, color='skyblue')
    axes[1, 0].bar(x, auc_scores, width, label='AUC Score', alpha=0.8, color='lightcoral')
    axes[1, 0].bar(x + width, cv_scores, width, label='CV Score', alpha=0.8, color='lightgreen')

    axes[1, 0].set_xlabel('Models', fontweight='bold')
    axes[1, 0].set_ylabel('Scores', fontweight='bold')
    axes[1, 0].set_title('Performance Metrics Comparison', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels
    for i, (acc, auc, cv) in enumerate(zip(test_accuracies, auc_scores, cv_scores)):
        axes[1, 0].text(i - width, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1, 0].text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1, 0].text(i + width, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=8)

    # 4. Calibration Curves (Bottom Right)
    for idx, (model_name, model) in enumerate(best_models.items()):
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
            axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 's-', color=colors[idx],
                            label=f'{model_name}')

    axes[1, 1].plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    axes[1, 1].set_xlabel('Mean Predicted Probability', fontweight='bold')
    axes[1, 1].set_ylabel('Fraction of Positives', fontweight='bold')
    axes[1, 1].set_title('Calibration Curves', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Comprehensive model comparison saved as 'comprehensive_model_comparison.png'")


def create_learning_curves_visualization(best_models, X_train, y_train):
    """Create learning curves visualization for all models"""
    print(f"\n📈 Creating Learning Curves Visualization...")

    plt.figure(figsize=(12, 8))

    for idx, (model_name, model) in enumerate(best_models.items()):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_scores_mean, 'o-', color=plt.cm.Set1(idx / len(best_models)),
                 label=f'{model_name} - Training')
        plt.plot(train_sizes, test_scores_mean, 'o--', color=plt.cm.Set1(idx / len(best_models)),
                 label=f'{model_name} - Cross-validation')

        # Fill between standard deviation areas
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color=plt.cm.Set1(idx / len(best_models)))
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color=plt.cm.Set1(idx / len(best_models)))

    plt.xlabel('Training Examples', fontweight='bold')
    plt.ylabel('Accuracy Score', fontweight='bold')
    plt.title('Learning Curves - All Models', fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Learning curves comparison saved as 'learning_curves_comparison.png'")


def create_confusion_matrices_visualization(best_models, X_test, y_test):
    """Create confusion matrices visualization for all models"""
    print(f"\n🎯 Creating Confusion Matrices Visualization...")

    n_models = len(best_models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (model_name, model) in enumerate(best_models.items()):
        if idx >= len(axes):
            break

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Plot confusion matrix
        im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[idx].set_title(f'{model_name}\n(Accuracy: {accuracy:.3f})', fontweight='bold')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")

        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Pred No\nDisease', 'Pred\nDisease'])
        axes[idx].set_yticklabels(['True No\nDisease', 'True\nDisease'])
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Confusion matrices comparison saved as 'confusion_matrices_comparison.png'")


def create_comprehensive_visualizations(best_models, X_test, y_test, X_train, y_train, selected_features, results):
    """Create comprehensive visualizations including ROC AUC"""
    print(f"\n📊 Creating All Comprehensive Visualizations...")

    # 1. Main comprehensive comparison (4-in-1)
    create_comprehensive_model_comparison(best_models, X_test, y_test, results, selected_features)

    # 2. Learning curves
    create_learning_curves_visualization(best_models, X_train, y_train)

    # 3. Feature importance (UPDATED - now returns importance data)
    importance_data = create_feature_importance_comprehensive(best_models, X_test, y_test, selected_features)

    # 4. Confusion matrices
    create_confusion_matrices_visualization(best_models, X_test, y_test)

    # 5. Combined probability distributions
    create_probability_distributions_combined(best_models, X_test, y_test)

    # 6. ROC AUC
    create_roc_auc_visualization(best_models, X_test, y_test, 'roc_auc_comparison.png')

    return importance_data


def print_correlation_matrix(df, selected_features, target_col='target'):
    """Print correlation matrix in terminal"""
    print("\n" + "=" * 60)
    print("📊 CORRELATION MATRIX")
    print("=" * 60)

    # Include target in correlation analysis
    features_to_plot = selected_features + [target_col]
    corr_matrix = df[features_to_plot].corr()

    print(tabulate(corr_matrix.round(3), headers=corr_matrix.columns, showindex=True, tablefmt='grid'))


def print_feature_distributions(df, selected_features, target_col='target'):
    """Print feature distribution statistics in terminal"""
    print("\n" + "=" * 60)
    print("📈 FEATURE DISTRIBUTION STATISTICS")
    print("=" * 60)

    # Select top features for clear visualization
    top_features = selected_features[:6] if len(selected_features) >= 6 else selected_features

    for feature in top_features:
        print(f"\n{feature}:")
        stats_df = df.groupby(target_col)[feature].agg(['mean', 'std', 'min', 'max']).round(3)
        print(tabulate(stats_df, headers='keys', tablefmt='grid'))


def print_learning_curves(models, X_train, y_train):
    """Print learning curve summary in terminal"""
    print("\n" + "=" * 60)
    print("📈 LEARNING CURVES SUMMARY")
    print("=" * 60)

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
            "High" if gap > 0.05 else "Medium" if gap > 0.02 else "Low"
        ])

    print(tabulate(learning_data,
                   headers=['Model', 'Final Train Acc', 'Final CV Acc', 'Gap', 'Overfitting Risk'],
                   tablefmt='grid'))


def print_confusion_matrices(best_models, X_test, y_test):
    """Print confusion matrices in terminal"""
    print("\n" + "=" * 60)
    print("🎯 CONFUSION MATRICES")
    print("=" * 60)

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{model_name} (Accuracy: {accuracy:.3f}):")
        print(tabulate(cm, headers=['Pred No Disease', 'Pred Disease'],
                       showindex=['True No Disease', 'True Disease'], tablefmt='grid'))


def print_roc_analysis(best_models, X_test, y_test):
    """Print ROC-AUC analysis in terminal"""
    print("\n" + "=" * 60)
    print("📈 ROC-AUC ANALYSIS")
    print("=" * 60)

    auc_data = []

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            auc_data.append([model_name, f"{auc_score:.4f}"])
        else:
            auc_data.append([model_name, "N/A"])

    print(tabulate(auc_data, headers=['Model', 'AUC Score'], tablefmt='grid'))


def print_prediction_probabilities(best_models, X_test, y_test):
    """Print prediction probability statistics in terminal"""
    print("\n" + "=" * 60)
    print("📊 PREDICTION PROBABILITY STATISTICS")
    print("=" * 60)

    for model_name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            stats_data = []

            for target_val in [0, 1]:
                mask = (y_test == target_val)
                probs = y_pred_proba[mask]
                stats_data.append([
                    'No Disease' if target_val == 0 else 'Disease',
                    f"{np.mean(probs):.3f}",
                    f"{np.std(probs):.3f}",
                    f"{np.min(probs):.3f}",
                    f"{np.max(probs):.3f}"
                ])

            print(f"\n{model_name}:")
            print(tabulate(stats_data, headers=['True Class', 'Mean', 'Std', 'Min', 'Max'], tablefmt='grid'))


def print_classification_reports(best_models, X_test, y_test):
    """Print classification reports for all models"""
    print("\n" + "=" * 80)
    print("📋 CLASSIFICATION REPORTS - ALL MODELS")
    print("=" * 80)

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        print(f"\n🎯 {model_name} Classification Report:")
        print(tabulate(report_df.round(4), headers='keys', tablefmt='grid'))


def enhanced_train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_features, df_encoded):
    print("\n" + "=" * 50)
    print("🤖 ENHANCED MODEL TRAINING & EVALUATION")
    print("=" * 50)

    # Create initial analyses
    print("\n📊 Creating initial analyses...")
    print_correlation_matrix(df_encoded, selected_features)
    print_feature_distributions(df_encoded, selected_features)

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

    for model_name, model in models.items():
        print(f"\n📈 Training {model_name}...")

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
            'Overfitting': overfitting_level
        })

    # Create comprehensive analyses
    print("\n📈 Creating comprehensive analyses...")
    print_learning_curves(models, X_train, y_train)
    print_confusion_matrices(best_models, X_test, y_test)
    print_roc_analysis(best_models, X_test, y_test)
    print_prediction_probabilities(best_models, X_test, y_test)
    print_classification_reports(best_models, X_test, y_test)

    # Create comprehensive visualizations
    importance_data = create_comprehensive_visualizations(best_models, X_test, y_test, X_train, y_train,
                                                          selected_features, results)

    # Print comprehensive feature importance analysis
    print_feature_importance_comprehensive(importance_data)

    # Display results
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

    return best_models, results, importance_data


def main():
    print("🚀 HEART DISEASE CLASSIFICATION PROJECT - ENHANCED VERSION")
    print("=" * 80)
    print("Selected Models: SVM (RBF), XGBoost, MLP, KNN, Voting Ensemble")
    print("Key Improvements:")
    print("  • Enhanced feature engineering with clinical features")
    print("  • Improved feature selection using multiple methods")
    print("  • Regularized models to prevent overfitting")
    print("  • Higher K values for KNN (15-30)")
    print("  • Ensemble methods for better generalization")
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
    print("\n📊 All analyses displayed in terminal above:")
    print("📈 COMPREHENSIVE VISUALIZATIONS SAVED:")
    print("   1. comprehensive_model_comparison.png - Main dashboard (4 plots)")
    print("   2. learning_curves_comparison.png - Learning curves")
    print("   3. comprehensive_feature_importance.png - Feature importance (ALL MODELS)")
    print("   4. confusion_matrices_comparison.png - Confusion matrices")
    print("   5. probability_distributions_combined.png - Combined probability distributions")
    print("   6. roc_auc_comparison.png - ROC curves")

    print("\n🔍 FEATURE IMPORTANCE METHODS USED:")
    for model_name, importance_df in importance_data.items():
        print(f"   - {model_name}: {importance_df['Method'].iloc[0]}")

    print("\n✅ Key improvements implemented:")
    print("   • KNN overfitting reduced with higher n_neighbors (15-30)")
    print("   • Enhanced feature engineering with clinical ratios")
    print("   • Improved feature selection using combined methods")
    print("   • Added ensemble methods for better generalization")
    print("   • Increased regularization across all models")


if __name__ == "__main__":
    main()