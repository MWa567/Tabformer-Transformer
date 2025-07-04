# -*- coding: utf-8 -*-
"""Transformer Downstream

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g2EbGE-QcFZxfvAGTg9HjoUuI5kTTg6e
"""

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
!pip install optuna

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys
sys.path.append('/content/drive/MyDrive/Freshman/UROP/Transformer')

import lgb_evaluation

import transformer_architecture

from importlib import reload

train_data = pd.read_csv('/content/drive/MyDrive/train_data_transformer.csv')
test_data = pd.read_csv('/content/drive/MyDrive/test_data_transformer.csv')

import torch

from transformer_architecture import Transformer

categorical = (15, 13, 32, 25, 61, 61, 8) # (category, Month, Day, Hour, Minute, Second, dayOfWeek)
numerical = 4 # (amt, merch_lat, merch_long, LatLong_Dist, hour_amt, category_amt)
num_layers = 6
max_len = 2049

model = Transformer(categorical, numerical, num_layers, max_len+1)
model_path = '/content/drive/My Drive/transformer_model.pth'
model.load_state_dict(torch.load(model_path))

# Get training data
train_num_trans_per_user = []
for user in train_data['cc_num'].unique():
  train_num_trans_per_user.append(train_data[train_data['cc_num'] == user].shape[0])

train_data_by_user = []
for user in train_data['cc_num'].unique():
  if train_data[train_data['cc_num'] == user].shape[0] > 1:
    user_data = train_data[train_data['cc_num'] == user].drop(['cc_num', 'is_fraud'], axis=1)
    user_data = np.append(user_data, np.zeros((max_len - user_data.shape[0], user_data.shape[1])), axis=0)
    train_data_by_user.append(torch.from_numpy(user_data))
train_data_by_user_tensor = torch.stack([user for user in train_data_by_user])

train_tgt_data = train_data_by_user_tensor

# Get testing data

test_num_trans_per_user = []
for user in test_data['cc_num'].unique():
  test_num_trans_per_user.append(test_data[test_data['cc_num'] == user].shape[0])

test_data_by_user = []
for user in test_data['cc_num'].unique():
  if test_data[test_data['cc_num'] == user].shape[0] > 1:
    test_user_data = test_data[test_data['cc_num'] == user].drop(['cc_num', 'is_fraud'], axis=1)
    test_user_data = np.append(test_user_data, np.zeros((max_len - test_user_data.shape[0], test_user_data.shape[1])), axis=0)
    test_data_by_user.append(torch.from_numpy(test_user_data))
test_data_by_user_tensor = torch.stack([user for user in test_data_by_user])

test_tgt_data = test_data_by_user_tensor

test_preds_cat = [] # List of 25 lists of probabilities for each of the 7 categorical features (1, 2048, 15), (1, 2048, 13), etc for all the transactions of a single user
test_preds_num = [] # List of 25 lists of predictions for each of the 4 numeric features (1, 2048, 1) for all the transactions of a single user
model.eval()
with torch.no_grad():
  for user in range(len(test_tgt_data)):
    pred_next_emb, preds_cat, preds_num = model(test_tgt_data[user, :-1, :])
    test_preds_cat.append(preds_cat)
    test_preds_num.append(preds_num)

import torch.nn.functional as F

preds_cat_tensor = []
for user in range(len(test_preds_cat)): # want to make a tensor with 15 columns and ~25 * 2048 rows for the categorical probabilities for all users
  preds_for_user = test_preds_cat[user]
  preds_for_user_concat = torch.cat([F.softmax(pred, dim=2).argmax(dim=2).cpu() for pred in preds_for_user], dim=0) # (2048 rows, 4 columns)
  preds_for_user_concat = torch.transpose(preds_for_user_concat, 0, 1)[:test_num_trans_per_user[user], :]
  preds_cat_tensor.append(preds_for_user_concat)

preds_cat_tensor = torch.concat(preds_cat_tensor, dim=0)
print(preds_cat_tensor.shape)

'''
preds_cat_tensor = []
for user in range(len(test_preds_cat)):
  preds_for_user = test_preds_cat[user] # list of 4 (1, 2048, 15) tensors
  for feature in range(len(preds_for_user)):
    transactions = F.softmax(preds_for_user[feature], dim=2).cpu() # (2048, 15) tensor of probabilities
    probabilities_per_user = []
    for trans in transactions:
      print(test_data[test_data['cc_num'] == test_data['cc_num'].unique()[user]])
      probability = trans[test_data[test_data['cc_num'] == test_data['cc_num'].unique()[user]][feature]]
      probabilities_per_user.append()
  preds_for_user_concat = torch.cat([F.softmax(pred, dim=2).cpu() for pred in preds_for_user], dim=0) # (2048 rows, 4 columns)
  preds_for_user_concat = torch.transpose(preds_for_user_concat, 0, 1)[:test_num_trans_per_user[user], :]
  preds_cat_tensor.append(preds_for_user_concat)

preds_cat_tensor = torch.concat(preds_cat_tensor, dim=0)
print(preds_cat_tensor.shape)
'''

preds_num_tensor = []
for user in range(len(test_preds_num)): # want to make a tensor with 4 columns and ~25 * 2048 rows for the numerical predictions for all users
  preds_for_user = test_preds_num[user]
  preds_for_user_concat = torch.cat([pred.squeeze(0)[:test_num_trans_per_user[user], :] for pred in preds_for_user], dim=1) # (2048 rows, 4 columns)
  preds_num_tensor.append(preds_for_user_concat)

preds_num_tensor = torch.concat(preds_num_tensor, dim=0)
print(preds_num_tensor.shape)

preds_tensor = torch.cat([preds_cat_tensor, preds_num_tensor], dim=1)
print(preds_tensor.shape)

preds_df = pd.DataFrame(preds_tensor.numpy(), columns=test_data.columns.drop(['cc_num', 'is_fraud']))

preds_df['is_fraud'] = test_data.reset_index(drop=True)['is_fraud']
preds_df['cc_num'] = test_data.reset_index(drop=True)['cc_num']

preds_df = preds_df.reindex(columns=test_data.columns)
test_data = test_data.drop(test_data.index[0])

preds_df.head()

test_data.head()

from sklearn.metrics.pairwise import cosine_similarity

test_array = test_data[['category', 'Month', 'Day', 'Hour', 'amt', 'merch_lat', 'merch_long', 'LatLong_Dist']].values
pred_array = preds_df[['category', 'Month', 'Day', 'Hour', 'amt', 'merch_lat', 'merch_long', 'LatLong_Dist']].values

similarity_matrix = cosine_similarity(test_array, pred_array)

cos_similarities = [similarity_matrix[i][i] for i in range(len(similarity_matrix))]

preds_df['cos_sim'] = cos_similarities

cos_sim_not_fraud = preds_df[preds_df['is_fraud'] == 0]['cos_sim']
cos_sim_fraud = preds_df[preds_df['is_fraud'] == 1]['cos_sim']

import statistics

print(statistics.mean(cos_sim_not_fraud), statistics.stdev(cos_sim_not_fraud))
print(statistics.mean(cos_sim_fraud), statistics.stdev(cos_sim_fraud))

plt.hist(cos_sim_not_fraud, bins=40)
plt.show()

plt.hist(cos_sim_fraud, bins=40)
plt.show()

test_differences = []
for user in range(len(test_preds_cat)):
  cat_differences = []
  num_differences = []
  test_tgt_cat = test_tgt_data[user, :, :][:, :7].long()
  test_tgt_num = test_tgt_data[user, :, :][:, 7:12].float()

  for i, pred in enumerate(test_preds_cat[user]):
    cat_differences_i = [x - y for x, y in zip(pred.argmax(dim=1).cpu().tolist()[1:test_num_trans_per_user[user]+1], test_tgt_cat[:, i].cpu().tolist()[1:])]
    cat_differences.append(cat_differences_i)
  for i, pred in enumerate(test_preds_num[user]):
    num_differences_i = [x - y for x, y in zip(test_preds_num[user][i].cpu().tolist()[1: test_num_trans_per_user[user]+1], test_tgt_num[:, i].cpu().tolist()[1:])]
    num_differences.append(num_differences_i)

  for i in range(len(cat_differences[0])):
    differences_trans = []
    for j in range(len(cat_differences)):
      differences_trans.append(cat_differences[j][i])
    for j in range(len(num_differences)):
      differences_trans.append(num_differences[j][i])
    test_differences.append(differences_trans)

test_differences = pd.DataFrame(test_differences)
test_differences = test_differences.rename(columns={0:'category', 1:'Month', 2:'Day', 3:'Hour', 4:'Minute', 5:'Second', 6:'dayOfWeek', 7:'amt', 8:'merch_lat', 9:'merch_long', 10:'LatLong_Dist'})

test_differences['is_fraud'] = test_data.reset_index(drop=True)['is_fraud']

test_differences.to_csv('/content/drive/MyDrive/test_differences_transformer.csv', index=False)

differences = preds_df[['category', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'dayOfWeek', 'amt', 'merch_lat', 'merch_long', 'LatLong_Dist']] - test_data[['category', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'dayOfWeek', 'amt', 'merch_lat', 'merch_long', 'LatLong_Dist']]
differences['is_fraud'] = test_data.reset_index(drop=True)['is_fraud']
differences['cc_num'] = test_data.reset_index(drop=True)['cc_num']
differences.drop(differences.index[0], inplace = True)
differences.drop(differences.index[-1], inplace = True)

differences.head()

train_data = differences.sample(frac=0.3, random_state=42)
test_data = differences.drop(train_data.index)

from sklearn.preprocessing import StandardScaler

for feature in ['category', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'dayOfWeek', 'amt', 'merch_lat', 'merch_long', 'LatLong_Dist']:
  train_data[feature] = StandardScaler().fit_transform(train_data[feature].values.reshape(-1, 1))
  test_data[feature] = StandardScaler().fit_transform(test_data[feature].values.reshape(-1, 1))

train_data["Class"] = train_data["is_fraud"]
train_data = train_data.drop(["is_fraud"], axis=1)
test_data["Class"] = test_data["is_fraud"]
test_data = test_data.drop(["is_fraud"], axis=1)

X_train, y_train, X_test, y_test = (
    train_data.drop(["Class"], axis=1),
    train_data["Class"],
    test_data.drop(["Class"], axis=1),
    test_data["Class"],
)

X_test, y_test = test_data.drop(["Class"], axis=1), test_data["Class"]

from collections import Counter
from lgb_evaluation import get_fraud_percentage

get_fraud_percentage(Counter(train_data["Class"]))
get_fraud_percentage(Counter(test_data["Class"]))

"""### To upsample minority class"""

# can duplicate minority class N times
from lgb_evaluation import duplicate_minority_class

new_dataset = duplicate_minority_class(train_data, N=10)
get_fraud_percentage(Counter(new_dataset["Class"]))

# can oversample minority class with weight
from imblearn.over_sampling import RandomOverSampler

num_fraud, denom_nonfraud = (
    10,
    90,
)  # <- this will result in fraction of fraud in new dataset = (num_fraud/(num_fraud+denom_nonfraud))
sampling_strategy = float(num_fraud / denom_nonfraud)
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
get_fraud_percentage(Counter(y_resampled))

"""### To train lightgbm"""

import optuna

def get_best_params(X_train, y_train):
    def objective(trial):
        # Define the parameter search space
        param = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.7, 0.9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        }

        # Prepare the dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Perform cross-validation with early stopping
        cv_results = lgb.cv(
            params=param,
            train_set=train_data,
            nfold=3,
            metrics=["auc"],
            # early_stopping_rounds=50,
            seed=42,
        )

        # Get the best AUC score from cross-validation (mean of validation AUC scores)
        return max(cv_results["valid auc-mean"])

    # Run the optimization process with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Retrieve the best parameters
    best_params = study.best_params

    return best_params

# Retrieve the best parameters
best_params = get_best_params(X_train, y_train)
print("Best parameters found:", best_params)

from lgb_evaluation import save_params

save_params(best_params, "best_params.json")

print(best_params)

"""### Evaluate LGB over multiple seeds"""

reload(lgb_evaluation)
from lgb_evaluation import evaluate_model_helper
from lgb_evaluation import calculate_partial_auc

NUM_SEEDS = 10  ## CHANGE THIS!!!
results_across_seeds = np.zeros((NUM_SEEDS, 12, 3))
partial_auc_across_seeds = []


for seed in range(NUM_SEEDS):
    model = lgb.LGBMClassifier(
        **best_params, n_estimators=100, random_state=seed, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    results = evaluate_model_helper(
        X_test, y_test, model, viz=True
    )  # <- SET TO FALSE TO NOT SEE THE CONFUSION MATRICES
    results_across_seeds[seed] = results

    partial_auc = calculate_partial_auc(y_test, model.predict_proba(X_test)[:, 1])
    partial_auc_across_seeds.append(partial_auc)  # partial auc at fpr threshold = 0.1%


columns = ["score_threshold", "fpr %", "recall %"]
averaged_results = np.mean(results_across_seeds, axis=0)
averaged_df = pd.DataFrame(averaged_results, columns=columns)

ave_partial_auc = np.mean(partial_auc_across_seeds)
ave_partial_auc_str = f"{ave_partial_auc:.4e}"

averaged_df

ave_partial_auc_str
