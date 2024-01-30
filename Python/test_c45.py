import pandas as pd
import numpy as np
from C45Classifier import C45
from main_run import load_dataset


def test_c45_fit():
    train_data_x = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    train_labels = pd.DataFrame({'class': [0, 1, 0]})
    features = ['feature1', 'feature2']
    feature_class = {'feature1': 'Continuous', 'feature2': 'Continuous'}

    c45 = C45(train_data_x, features, train_labels, feature_class)
    c45.fit()

    assert c45.root is not None


def test_c45_large():
    num = 50000
    feature1 = np.random.rand(num)
    feature2 = np.random.randint(0, 100, num)
    classes = np.random.randint(0, 2, num)

    train_data_x = pd.DataFrame({'feature1': feature1, 'feature2': feature2})
    train_labels = pd.DataFrame({'class': classes})
    features = ['feature1', 'feature2']
    feature_class = {'feature1': 'Continuous', 'feature2': 'Continuous'}

    c45 = C45(train_data_x, features, train_labels, feature_class, percentage=5, min_ins=1000)
    c45.fit()
    assert c45.root is not None


def test_c45_missing_value():
    X = load_dataset(45)

    assert X[0].isna().any().any() == False


def test_c45_predict():

    train_data_x = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    train_labels = pd.DataFrame({'class': [0, 1, 0]})
    features = ['feature1', 'feature2']
    feature_class = {'feature1': 'Continuous', 'feature2': 'Continuous'}

    c45 = C45(train_data_x, features, train_labels, feature_class)
    c45.fit()

    test_data = pd.DataFrame({'feature1': [2], 'feature2': [5]})
    prediction = c45.predict(test_data)
    assert len(prediction) == 1
    assert prediction.iloc[0] in [0, 1]
