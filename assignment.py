#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(target_column, **kwargs):
    task_instance = kwargs['ti']
    data = task_instance.xcom_pull(task_ids="read_csv")
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train.to_dict(), X_test.to_dict(), y_train.to_dict(), y_test.to_dict()

def train_linear_regression( **kwargs):
    task_instance = kwargs['ti']
    X_train_dict, _, y_train_dict, _ = task_instance.xcom_pull(task_ids="preprocess_data")
    X_train = pd.DataFrame.from_dict(X_train_dict)
    y_train = pd.Series(y_train_dict)
    model = LinearRegression()
    model.fit(X_train, y_train)
    plt.scatter(X, y,  color='black')
    plt.title('linear regression')
    plt.xlabel('saar')
    plt.ylabel('HPI')
    plt.xticks(())
    plt.yticks(())
    plt.plot(X_train, model.predict(X_train), color='red',linewidth=3)
    plt.show()
    model_filepath = "/mnt/c/c/users/jerry/airflowhome/linear_regression_model.pkl"
    joblib.dump(model, model_filepath)
    return model_filepath

def test_model(**kwargs):
    
    task_instance = kwargs['ti']
    
    model_filepath = task_instance.xcom_pull(task_ids="train_linear_regression")
    
    model = joblib.load(model_filepath)
    
    _, X_test_dict, _, y_test_dict = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_test = pd.DataFrame.from_dict(X_test_dict)
    y_test = pd.Series(y_test_dict)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = model.score(X, y)
    adj_r_squared = r_squared - (1 - r_squared) * (X.shape[1] / (X.shape[0] - X.shape[1] - 1))
    p_value = f_regression(X, y)
    return mse, r_squared, adj_r_squared, p_value

default_args = {
    'owner': 'jerry',
    'start_date': datetime(2023, 6, 17),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    dag_id='linear_regression_pipeline',
    default_args=default_args,
    description='read csv, train and test linear regression model',
    schedule_interval=timedelta(days=1),
)

file_path = "data.csv"
target_column = "HPI"

task1 = PythonOperator(
    task_id="read_csv",
    python_callable=read_csv,
    op_args=[file_path],
    dag=dag,
)

task2 = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    op_args=[target_column],
    provide_context=True,
    dag=dag,
)

task3 = PythonOperator(
    task_id="train_linear_regression",
    python_callable=train_linear_regression,
    provide_context=True,
    dag=dag,
)

task4 = t4 = PythonOperator(
    task_id="test_model",
    python_callable=test_model,
    provide_context=True,
    dag=dag,
)

task1 >> task2 >> task3 >> task4

