# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--output-model-name', dest='output_model_name', type=str, help='output model name')
parser.add_argument('--feature-list-names', dest='feature_list_names', type=str, help='list of input features')
parser.add_argument('--categorical-feature-list-names', dest='categorical_feature_list_names', type=str, help='list of input features')
parser.add_argument('--target', dest='target', type=str, help='target column name')
args = parser.parse_args()

# 1. Load training and testing data
run = Run.get_context()
input_data_train = run.input_datasets['output_split_train']
input_data_test  = run.input_datasets['output_split_test']
input_df_train = input_data_train.to_pandas_dataframe().drop('CUSTOMER_ID', axis=1)
input_df_test  = input_data_test.to_pandas_dataframe().drop('CUSTOMER_ID', axis=1)
feature_columns = args.feature_list_names.split(",")
categorical_feature_columns = args.categorical_feature_list_names.split(",")
target_column = args.target

X_train, y_train = input_df_train[feature_columns], input_df_train[target_column]
X_test, y_test = input_df_test[feature_columns], input_df_test[target_column]

# 2. Handle categorical feature for lgbm
for column in categorical_feature_columns:
    X_train[column] = X_train[column].astype('category')
    X_test[column] = X_test[column].astype('category')

run.log('Input columns: ', ', '.join(list(input_df_train.columns)))

global hyper_params
hyper_params = {
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "sub_feature": 0.7,
    "num_leaves": 60,
    "min_data": 100,
    "min_hessian": 1,
    "verbose": 0
}

# 3. Train model
def train_eval(X_train, y_train, X_test, y_test):
    train_data = lightgbm.Dataset(X_train, label=y_train)
    valid_data = lightgbm.Dataset(X_test, label=y_test, free_raw_data=False)
    model = lightgbm.train(
        hyper_params,
        train_data,
        valid_sets=valid_data,
        num_boost_round=500,
        early_stopping_rounds=20)
    return model

print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_name = args.output_model_name if args.output_model_name is not None else 'model'
model_file = os.path.join('outputs', '%s.pkl'%(model_name))
model = train_eval(X_train, y_train, X_test, y_test)
joblib.dump(value=model, filename=model_file)
print(" [Successful] save model in ", model_file)

# 4. Evaluate model
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_th05 = [1 if p >= 0.5 else 0 for p in y_pred]
    cls_rp = classification_report(y_test, y_pred_th05, output_dict = True)
    cls_rp = pd.DataFrame(cls_rp).transpose()
    metrics = dict(cls_rp.loc['weighted avg'])
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1-score']
    auc = roc_auc_score(y_test, y_pred)
    return precision, recall, f1, auc

test_precision, test_recall, test_f1, test_auc = get_metrics(model, X_test, y_test)
train_precision, train_recall, train_f1, train_auc = get_metrics(model, X_train, y_train)
# 5. Save the trained model in the outputs folder
# Register the model
print('Registering model...')
Model.register(
    workspace=run.experiment.workspace,
    model_path = model_file,
    model_name = model_name,
    tags={'Training context':'Pipeline'},
    properties={
        'data.train.shape': X_train.shape[0],
        'data.test.shape': X_test.shape[0],
        'data.features': feature_columns,
        'data.target_column': target_column,
        'model.algorithm':  type(model).__name__,
        'model.model_params': hyper_params,
        'evaluation.test.precision': test_precision,
        'evaluation.test.recall': test_recall,
        'evaluation.test.f1': test_f1,
        'evaluation.test.auc': test_auc,
        'evaluation.train.precision': train_precision,
        'evaluation.train.recall': train_recall,
        'evaluation.train.f1': train_f1,
        'evaluation.train.auc': train_auc,
    }
)

run.complete()