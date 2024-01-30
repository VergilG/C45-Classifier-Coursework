from C45Classifier import C45
import sklearn.model_selection as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ucimlrepo import fetch_ucirepo
import time
import tracemalloc
import pandas as pd


def load_dataset(data_id_in_uci: int):
    # Load dataset
    raw_data = fetch_ucirepo(id=data_id_in_uci)

    name = raw_data.metadata['name']
    num_instances = raw_data.metadata['num_instances']
    num_features = raw_data.metadata['num_features']

    # Split data
    X = raw_data.data.features
    y = raw_data.data.targets
    y.columns = ['class']

    features = raw_data.variables.name.tolist()[:-1]
    classes = raw_data.variables.type.tolist()[:-1]

    features_classes = {}
    for i in range(len(features)):
        features_classes[features[i]] = classes[i]

    # Deal with missing values.
    if_missed_features = raw_data.variables.missing_values.tolist()
    for i in range(len(if_missed_features)):
        if if_missed_features[i] == 'yes':
            mode_value = X[features[i]].mode()[0]
            X.iloc[:, i].fillna(mode_value, inplace=True)

    return X, y, features, features_classes, num_instances, num_features, name


def run_model(ID, p: int, train_size, min):
    X, y, features, features_classes, num_instances, num_features, name = load_dataset(ID)
    train_data_size = train_size
    X_train, X_test, y_train, y_test = sk.train_test_split(X, y, train_size=train_data_size, random_state=97)

    # Use the training dataset train the decision tree.
    tracemalloc.start()
    start_time = time.time()
    c45_model = C45(X_train, features, y_train, features_classes, percentage=p, min_ins=min)
    c45_model.fit()
    run_time = round((time.time() - start_time) * 1000, 2)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predict_result = c45_model.predict(X_test)
    y_test_series = y_test.iloc[:, 0]

    Accuracy_model = (predict_result == y_test_series).mean()

    Precision_model = 0.0
    labels = y.iloc[:, 0].unique().tolist()
    for label in labels:
        true_predictions = ((predict_result == label) & (y_test_series == label)).sum()
        predictions = (predict_result == label).sum()
        if predictions > 0:
            Precision_model += true_predictions / predictions
    Precision_model = Precision_model / len(labels)

    Recall_model = 0.0
    for label in labels:
        true_predictions = ((predict_result == label) & (y_test_series == label)).sum()
        actual_labels = (y_test_series == label).sum()
        if actual_labels > 0:
            Recall_model += true_predictions / actual_labels
    Recall_model = Recall_model / len(labels)

    F1_Score_model = 2 * (Precision_model * Recall_model) / (Precision_model + Recall_model)

    result_dict = {'Name': name,
                   'No_features': num_features,
                   'No_instances': num_instances,
                   'Accuracy': Accuracy_model,
                   'Precision': Precision_model,
                   'Recall': Recall_model,
                   'F1_Score': F1_Score_model,
                   'Split_points': c45_model.percentage,
                   'Train_set_size': train_data_size,
                   'Min_instances_node': c45_model.Max,
                   'Run_time': run_time,
                   'Memory_use': current / 10 ** 6}

    return result_dict


def run_sklearn_tree(data_id, train_size):
    X, y, features, features_classes, num_instances, num_features, name = load_dataset(data_id)
    train_data_size = train_size
    X_train, X_test, y_train, y_test = sk.train_test_split(X, y, train_size=train_data_size, random_state=97)

    tracemalloc.start()
    start_time = time.time()
    sk_tree = DecisionTreeClassifier()  # Create decision tree.
    sk_tree.fit(X_train, y_train)  # Train the model.
    run_time = round((time.time() - start_time)/1000, 6)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred = sk_tree.predict(X_test)  # Predict.

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='weighted')

    recall = recall_score(y_test, y_pred, average='weighted')

    f1 = f1_score(y_test, y_pred, average='weighted')

    result_dict = {'Name': name,
                   'No_features': num_features,
                   'No_instances': num_instances,
                   'Accuracy': accuracy,
                   'Precision': precision,
                   'Recall': recall,
                   'F1_Score': f1,
                   'Split_points': 'sklearn',
                   'Train_set_size': train_data_size,
                   'Min_instances_node': 'sklearn',
                   'Run_time': run_time,
                   'Memory_use': current / 10 ** 6}

    return result_dict


if __name__ == '__main__':
    data_id_list = [53, 602, 850, 545, 159]  # All datasets used.
    split_points = [3, 4, 5, 6, 7, 8, 9, 10]
    train_size = [0.5, 0.6, 0.7, 0.8, 0.9]
    min_instances_in_node = [50, 100, 200, 500, 1000]

    csv_list = []

    for ID in data_id_list:
        for p in split_points:
            for size in train_size:
                for i in min_instances_in_node:
                    csv_list.append(run_model(ID, p, size, i))

    for ID in data_id_list:
        for size in train_size:
            csv_list.append(run_sklearn_tree(ID, size))

    csv_df = pd.DataFrame(csv_list)  # All results ready for analysis in R.

    csv_df.to_csv('model_results.csv', index=False)  # Save results into a csv file.
