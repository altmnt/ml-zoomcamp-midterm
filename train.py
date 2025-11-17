#!/usr/bin/env python
# coding: utf-8

# # Midterm Project - Pet Preference Prediction Using Machine Learning.
#


import csv
from collections import defaultdict

import pandas as pd
import numpy as np
import pyreadstat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction import DictVectorizer



SEED = 1


df_src, meta = pyreadstat.read_sav("animals.sav", apply_value_formats=True) # SPSS file format

with open("to_translate.csv", newline='', encoding='utf-8') as fo: # original
    with open("to_translate_with_labels.csv", newline='', encoding='utf-8') as ft: # translation
        orig_reader = csv.DictReader(fo)
        translation_reader = csv.DictReader(ft)
        labels = list(zip(orig_reader, translation_reader))

lbl_lookup = defaultdict(dict)
for ru,en in labels:
    lbl_lookup[ru['column']][ru['value']]=en['label']
list(lbl_lookup.items())[0]


categorical = [c for c,dtype in df_src.dtypes.items() if dtype == 'category']
df_translated = df_src[[col for col in df_src.columns if col not in categorical]].copy()
for cat in categorical:
    df_translated[cat] = df_src[cat].map(lambda i: lbl_lookup[cat][i] if i is not np.nan else np.nan)

cols = [col for col in df_translated.columns if col.startswith('dzh1')]
def get_target(row):
    C = any('cat' in row[col] for col in cols if row[col] is not np.nan)
    D = any('dog' in row[col] for col in cols if row[col] is not np.nan)
    if C and not D:
        return 'Cat'
    elif D and not C:
        return 'Dog'
    elif D and C:
        return 'Both'
    else:
        return 'Neither'


df_translated['y'] = df_translated.apply(get_target, axis=1)
df_translated.drop(columns = cols, inplace = True)
del df_translated['weight1']
del df_translated['ID']
del df_translated['FO']
dzh2_cols = [col for col in df_translated.columns if col.startswith('dzh2')]
df_translated.drop(columns=dzh2_cols, inplace=True)
df_translated.drop(columns=['dzh3'], inplace=True)
df_translated.drop(columns=['dzh4'], inplace=True)
dzh5_cols = [col for col in df_translated.columns if col.startswith('dzh5')]
df_translated.drop(columns=dzh5_cols, inplace=True)
# SEX Record the respondent's gender WITHOUT ASKING:
# AGE How old are you (full years)?
# TIP Type of settlement
# TV. Let's talk a bit about you. Tell me, please, do you watch TV or not? If you do, how often?
# d1. Tell me, please, do you use the internet? If yes, how often?
# EDU. What is your education level?
# DOHOD-0. How would you assess the current financial situation of your family - you and your relatives permanently living with you?
# PROF_1. Tell me, please, what is your main occupation at the moment? I will read out the answer options, and you choose the one that suits you.
# PROF_2. In our country there are organizations of different types. Now I will read out several options, and you tell me which one best corresponds to the organization/enterprise where you work.
# PROF_3. Tell me, what industry or field of activity does the organization where you work belong to?
# ```
df_translated[['PROF2','PROF3']] = df_translated[['PROF2','PROF3']].fillna('unknown')
df = df_translated

numerical_cols = ['AGE']
df_train_val, df_test  = train_test_split(df, test_size=.2, random_state=SEED)
df_train, df_val  = train_test_split(df_train_val, test_size=.25, random_state=SEED) # 0.25 x 0.8 = 0.2

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train = df_train.y
df_train.drop(columns=['y'], inplace=True)

y_test = df_test.y
df_test.drop(columns=['y'], inplace=True)

y_val = df_val.y
df_val.drop(columns=['y'], inplace=True)

dv = DictVectorizer(sparse=False)
dv.fit(df_train.to_dict(orient='records'))

target_labels = ['Neither','Cat','Both','Dog']

X_train = dv.transform(df_train.to_dict(orient='records'))
X_val = dv.transform(df_val.to_dict(orient='records'))
X_test = dv.transform(df_test.to_dict(orient='records'))


def xgb_classification_metrics(xgb_model, dmatrix, y_true, label_encoder, show = True):
    """Print metrics for XGBoost model trained with xgb.train()"""
    # Get predictions (returns probability matrix)
    y_pred_proba = xgb_model.predict(dmatrix)

    # Get predicted classes
    y_pred_encoded = y_pred_proba.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    aucs = []
    # Calculate ROC AUC for each class
    # IMPORTANT: label_encoder encodes alphabetically, so we need to match indices correctly
    for lbl in target_labels:
        y_binary = (y_true == lbl).astype(int)
        # Get the encoded index for this label
        lbl_idx = label_encoder.transform([lbl])[0]
        auc = roc_auc_score(y_binary, y_pred_proba[:, lbl_idx])
        if show:
            print(f"ROC AUC for class {lbl}: {auc:.4f}")
        aucs.append(auc)

    # Calculate accuracy
    accuracy = (y_pred == y_true).mean()
    if show:
        print(f"Overall Accuracy: {accuracy:.4f}")
    return np.average(aucs)

import xgboost as xgb

def train_xgboost_model(X_train, y_train, X_val, y_val, seed=1):
    """
    Train XGBoost model with optimal parameters.

    Parameters:
    -----------
    X_train : array-like
        Training features (already transformed by DictVectorizer)
    y_train : array-like
        Training labels (string labels: 'Neither', 'Cat', 'Both', 'Dog')
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    model : xgboost.Booster
        Trained XGBoost model
    label_encoder : LabelEncoder
        Fitted label encoder (needed for predictions)
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Best parameters from hyperparameter tuning
    xgb_parameters = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'seed': seed,
        'eta': 0.001,  # Best learning rate
        'max_depth': 5,  # Best tree depth
        'min_child_weight': 1,
        'subsample': 0.8,
        'nthread': 8,
        'verbosity': 0,
        'silent': 1
    }

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dval = xgb.DMatrix(X_val, label=y_val_encoded)

    # Train model
    model = xgb.train(
        params=xgb_parameters,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        verbose_eval=False
    )

    return model, label_encoder


def predict_xgboost(model, label_encoder, X, target_labels=['Neither', 'Cat', 'Both', 'Dog']):
    """
    Make predictions using trained XGBoost model.

    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    label_encoder : LabelEncoder
        Fitted label encoder from training
    X : array-like
        Features (already transformed by DictVectorizer)
    target_labels : list
        List of class labels in order

    Returns:
    --------
    predictions : array
        Predicted class labels
    probabilities : array
        Predicted probabilities for each class
    """
    import xgboost as xgb

    # Create DMatrix
    dmatrix = xgb.DMatrix(X)

    # Get probability predictions
    y_pred_proba = model.predict(dmatrix)

    # Get class predictions
    y_pred_encoded = y_pred_proba.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    return y_pred, y_pred_proba


# ### Test the train and predict functions

# In[35]:


# Train model using the functions
print("Training XGBoost model with best parameters...")
final_xgb_model, final_le = train_xgboost_model(X_train, y_train, X_val, y_val, seed=SEED)
print("Model trained successfully")


# In[36]:


# Test predictions
print("Testing predict function on test set...")
y_pred_test, y_pred_proba_test = predict_xgboost(final_xgb_model, final_le, X_test)

print(f"\nPrediction shape: {y_pred_test.shape}")
print(f"Probability shape: {y_pred_proba_test.shape}")
print(f"\nFirst 5 predictions: {y_pred_test[:5]}")
print(f"First 5 actual labels: {y_test[:5].values}")

import pickle
# Save the trained XGBoost model
output_file = 'pet_preference_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((final_xgb_model, final_le, dv, target_labels), f_out)

print(f"Model saved to {output_file}")
print(f"\nSaved objects:")
print(f"  1. final_xgb_model: Trained XGBoost model")
print(f"  2. final_le: Label encoder for target classes")
print(f"  3. dv: DictVectorizer for feature transformation")
print(f"  4. target_labels: List of class labels {target_labels}")


