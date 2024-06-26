import random
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import shap
# import tensorflow.keras as keras
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc as auc_score
from tensorflow import keras


def get_model(model_name):
    params = {
        "n_estimators": 108,
        "max_depth": 8,
        "learning_rate": 0.05,
        "booster": "dart",
        "n_jobs": 80,
        "gamma": 0.47,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "base_score": 0.5,
        "random_state": 4,
        "num_parallel_tree": 1,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "eta": 0.15
    }

    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier()
    elif model_name == 'SVM':
        model = SVC(probability=True)
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'XGBoost':
        model = XGBClassifier(**params)
    elif model_name == 'LightGBM':
        model = LGBMClassifier()
    elif model_name == 'CatBoost':
        model = CatBoostClassifier()
    elif model_name == 'SGD':
        model = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, tol=1e-3)
    return model


def train_and_evaluate_model_with_cv(input_file, output_csv, model_name, X_list, y_list):
    data = pd.read_csv(input_file)

    if model_name in ['RandomForest', 'LogisticRegression', 'MLP']:
        data.dropna(inplace=True)

    X = data[X_list]
    y = data[y_list]

    kf = KFold(n_splits=5, shuffle=True)

    results_cv = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = get_model(model_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_prob)
        aps = average_precision_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        npv = confusion_matrix(y_test, y_pred)[0, 0] / (
                confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[1, 0])
        fpr = 1 - specificity
        fnr = 1 - recall
        fdr = 1 - precision
        f1 = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred_prob)

        results = {
            'AUC': auc,
            'APS': aps,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'Specificity': specificity,
            'NPV': npv,
            'FPR': fpr,
            'FNR': fnr,
            'FDR': fdr,
            'F1 Score': f1,
            'Brier Score': brier
        }
        results_cv.append(results)

    df_cv = pd.DataFrame(results_cv)
    print("Cross-Validation Results:")
    print(df_cv)
    df_cv.to_csv(output_csv, index=False)

def train_model_search_param(input_file, model_name, X_list, y_list, result_file, auc_plot_file, pr_plot_file):

    data = pd.read_csv(input_file)

    X = data[X_list]
    y = data[y_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if model_name == 'LogisticRegression':
        model = LogisticRegression()
        param_distributions = {
            'penalty': ['l1', 'l2'],  
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  
            'solver': ['liblinear']  
        }
    elif model_name == 'RandomForest':
        model = RandomForestClassifier()
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],  
            'max_depth': [None, 10, 20, 30, 40, 50],  
            'min_samples_split': [2, 5, 10, 15, 20],  
            'min_samples_leaf': [1, 2, 4, 6, 8]  
        }
    elif model_name == 'SVM':
        model = SVC(probability=True)
        param_distributions = {
            'C': [0.01, 0.1, 1, 10, 100, 1000],  
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'gamma': ['scale', 'auto']  
        }
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
        param_distributions = {
            'n_neighbors': [3, 5, 7, 9, 11],  
            'weights': ['uniform', 'distance'],  
            'metric': ['euclidean', 'manhattan', 'minkowski']  
        }
    elif model_name == 'XGBoost':
        model = XGBClassifier()
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],  
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]  
        }
    elif model_name == 'LightGBM':
        model = LGBMClassifier()
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],  
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  
            'num_leaves': [7, 10, 20, 31, 50, 100, 150, 200]  
        }
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(verbose=0)
        param_distributions = {
            'iterations': [100, 200, 300, 400, 500],  
            'depth': [3, 4, 5, 6, 7, 8, 9, 10],  
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]  
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    grid_search = GridSearchCV(estimator=model, param_grid=param_distributions, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, y)


    best_model = grid_search.best_estimator_

    print("Best parameters found: ", grid_search.best_params_)

    '''

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=5,
                                       scoring='roc_auc', n_jobs=-1, random_state=random_num)
    random_search.fit(X_resampled, y_resampled)


    best_model = random_search.best_estimator_
    '''

    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    aps = average_precision_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    fpr = 1 - specificity
    fnr = 1 - recall
    fdr = 1 - precision
    brier = brier_score_loss(y_test, y_pred_prob)

    results = {
        'Metric': ['AUC', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F1 Score'],
        'Value': [auc, accuracy, recall, precision, specificity, f1]
    }
    df = pd.DataFrame(results)
    # df.to_csv(result_file, index=False)
    print(df)


    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    # plt.savefig(auc_plot_file)
    plt.show()


    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # plt.savefig(pr_plot_file)
    plt.show()


def kerasMLP(input_file, X_list, y_list):
    data = pd.read_csv(input_file)
    data_cleaned = data.dropna()

    X = data_cleaned[X_list]
    y = data_cleaned[y_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

    model = keras.models.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    adam_optimizer = keras.optimizers.Adam()
    model.compile(optimizer=adam_optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_resampled, y_train_resampled,
                        epochs=200, batch_size='auto',
                        validation_split=0.2,
                        shuffle=True)

    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # 早停法
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)  # 学习率衰减

    # keras.models.save_model(model, "../cmm_200feature/model/deep.h5")

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    performance_metrics = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                                        'Value': [accuracy, precision, recall, f1, auc]})
    print(performance_metrics)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


if __name__ == '__main__':
    input_file = "../data/final.csv"
    # LogisticRegression、RandomForest、MLP、LightGBM、XGBoost、SVM、CatBoost
    model_name = "RandomForest"
    df = pd.read_csv(input_file)
    X_list = ['original_shape_Sphericity', 'wavelet-LLH_firstorder_Range.2', 'wavelet-LLH_glcm_Autocorrelation.2',
              'wavelet-LLH_glcm_ClusterShade.2', 'wavelet-LLH_glcm_ClusterShade.5', 'wavelet-LLH_glcm_Contrast.2',
              'wavelet-LLH_glcm_DifferenceVariance.3', 'wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis.2',
              'wavelet-LLH_glrlm_GrayLevelVariance.7', 'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis.2']
    # X_list = ['original_shape_Sphericity', 'wavelet-LLH_firstorder_Range.2', 'wavelet-LLH_glcm_Autocorrelation.2',
    #           'wavelet-LLH_glcm_ClusterShade.2', 'wavelet-LLH_glcm_ClusterShade.5', 'wavelet-LLH_glcm_Contrast.2',
    #           'wavelet-LLH_glcm_DifferenceVariance.3', 'wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis.2',
    #           'wavelet-LLH_glrlm_GrayLevelVariance.7', 'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis.2', 'age', 'TTP',
    #           'peritumor edema', 'IMPCs']
    # X_list = ['age', 'TTP', 'peritumor edema', 'IMPCs']
    y_list = ['group']
    output_csv = '../data/kfold/2/omics.csv'
    train_and_evaluate_model_with_cv(input_file, output_csv, model_name, X_list, y_list)
    kerasMLP(input_file, X_list, y_list)
