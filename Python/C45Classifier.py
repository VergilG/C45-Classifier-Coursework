# Import library
import pandas as pd
import numpy as np
import math


class DecisionTreeNode:
    def __init__(self, feature, value=None, split=None):
        self.feature = feature  # Feature index or colum name.
        self.value = value  # Internal and root node is none.
        self.split = split  # Split points to split children nodes.
        self.child = {}  # A dict is suitable for both continuous and categorical features.

    def is_leaf_node(self):
        return self.value is not None

    def add_children_node(self, feature_value, node):
        self.child[feature_value] = node


class C45:
    def __init__(self, train_data_x, train_features: list, train_labels, feature_class: dict, percentage=5, min_ins=1000):
        self.MinGainRatio = 0.1  # A threshold for stopping the growing of decision tree.
        self.Max = train_data_x.shape[0]/min_ins  # Another threshold.
        self.train_data = pd.concat([train_data_x, train_labels], axis=1)
        self.features = train_features  # All unique features in train set.
        self.features_class = feature_class  # Continuous and Integer will be considered as numerical feature.
        self.root = None
        self.infoE = self.calculate_information_entropy(train_labels)  # The information entropy of whole train set.
        self.split_point = {2: [0.5],
                            3: [0.33, 0.67],
                            4: [0.25, 0.5, 0.75],
                            5: [0.2, 0.4, 0.6, 0.8],
                            6: [0.16, 0.32, 0.48, 0.64, 0.8],
                            7: [0.14, 0.28, 0.42, 0.56, 0.70, 0.84],
                            8: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
                            9: [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88],
                            10: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}  # Different split points for analysis.
        self.percentage = percentage  # Point to different split points.

    def calculate_information_entropy(self, d: pd.Series):
        try:
            unique_labels = d.value_counts(normalize=True)
            IE = 0  # Information Entropy

            for p in unique_labels:
                IE -= p * math.log(p, 2)

            return IE
        except Exception as e:
            print(f"Error in calculating Information Entropy: {e}")

    def find_best_feature(self, data, features):
        try:
            GainRatio = -float('inf')  # Initial a most large Gain Ratio, can be smaller than any common Gain Ratio.
            feature_selected = ''

            for feature in features:
                # Calculate the Solit Info.
                SplitInfo = self.calculate_information_entropy(data[feature])
                if SplitInfo == 0:
                    return feature, 0

                # Two kinds of features with different methods.
                if self.features_class[feature] == 'Continuous' or self.features_class[feature] == 'Integer':
                    subsets = self.split(self.features_class[feature], data, self.calculate_split_continue(feature, data), feature)
                    if subsets[0].shape[0] == 0 or subsets[1].shape[0] == 0:
                        continue
                    else:
                        IE = 0
                        for subset in subsets:
                            IE += self.calculate_information_entropy(subset) * (len(subset) / data.shape[0])
                else:
                    labels = data[feature].unique()
                    IE = 0

                    for label in labels:
                        subset = data[data[feature] == label].iloc[:, -1]
                        IE += self.calculate_information_entropy(subset) * (len(subset) / data.shape[0])

                Gain = self.infoE - IE

                if (Gain / SplitInfo) > GainRatio:
                    GainRatio = Gain / SplitInfo
                    feature_selected = feature

                if feature_selected == '':
                    GainRatio = 0
                    feature_selected = features[0]

            return feature_selected, GainRatio
        except Exception as e:
            print(f"Error in finding split feature: {e}")

    def calculate_split_continue(self, feature, data):
        try:
            # Use quintiles.
            quintiles = data[feature].quantile(self.split_point[self.percentage])

            best_quintile = min(data[feature])
            best_weighted_variance = float('inf')
            for quintile in quintiles:
                subset1 = data[data[feature] >= quintile]
                subset2 = data[data[feature] < quintile]

                variance1 = np.var(subset1[feature])
                variance2 = np.var(subset2[feature])

                weighted_variance = (len(subset1) / len(data)) * variance1 + (len(subset2) / len(data)) * variance2

                if weighted_variance < best_weighted_variance:
                    best_weighted_variance = weighted_variance
                    best_quintile = quintile

            return best_quintile
        except KeyError:
            print("Feature do not exist.")
        except Exception as e:
            print(f"Error in calculating split point when feature class is continuous: {e}")

    def calculate_split_classes(self, feature, data):
        try:
            # Not numerical.
            split_points = data[feature].unique().tolist()
            return split_points
        except KeyError:
            print("Feature do not exist.")
        except Exception as e:
            print(f"Error in calculating split point when feature class is *not* continuous: {e}")

    def split(self, feature_class, data, split, split_feature):
        try:
            split_data = []

            if feature_class == 'Continuous' or feature_class == 'Integer':
                split_data.append(data[data[split_feature] < split])
                split_data.append(data[data[split_feature] >= split])
            else:
                for point in split:
                    split_data.append(data[data[feature_class] == point])

            return split_data
        except KeyError:
            print("Feature do not exist.")
        except Exception as e:
            print(f"Error in calculating split subsets: {e}")

    def build_tree(self, data, features):
        try:
            split_feature, GainRatio = self.find_best_feature(data.iloc[:, :-1], features)

            features = [x for x in features if x != split_feature]

            if 0 < GainRatio < self.MinGainRatio or features == [] or data.shape[0] < self.Max:
                value = data['class'].value_counts().idxmax()
                return DecisionTreeNode(split_feature, value=value)

            if self.features_class[split_feature] == 'Continuous':
                node = DecisionTreeNode(split_feature, split=self.calculate_split_continue(split_feature, data))
                subsets = self.split('Continuous', data, node.split, split_feature)

                node.add_children_node('left', self.build_tree(subsets[0], features))
                node.add_children_node('right', self.build_tree(subsets[1], features))

                return node
            elif self.features_class[split_feature] == 'Integer':
                node = DecisionTreeNode(split_feature, split=self.calculate_split_continue(split_feature, data))
                subsets = self.split('Integer', data, node.split, split_feature)

                node.add_children_node('left', self.build_tree(subsets[0], features))
                node.add_children_node('right', self.build_tree(subsets[1], features))

                return node
            else:
                node = DecisionTreeNode(split_feature, split=self.calculate_split_classes(split_feature, data))
                splits = node.split.copy()
                splits.append('not_in_split')
                for split in splits:
                    subset = data[data[split_feature] == split]
                    node.add_children_node(split, self.build_tree(subset, features))

                return node
        except KeyError:
            print("Feature do not exist.")
        except Exception as e:
            print(f"Error in building tree: {e}")

    def run_tree(self, data, node: DecisionTreeNode):
        try:
            if node.is_leaf_node():
                return node.value

            feature_value = data[node.feature]

            if isinstance(node.split, list):
                if feature_value in node.split:
                    return self.run_tree(data, node.child[feature_value])
                else:
                    feature_value = 'not_in_split'
                    return self.run_tree(data, node.child[feature_value])
            else:
                if feature_value < node.split:
                    return self.run_tree(data, node.child['left'])
                else:
                    return self.run_tree(data, node.child['right'])
        except KeyError:
            print("Feature do not exist.")
        except Exception as e:
            print(f"Error in running the model: {e}")

    def fit(self):
        self.root = self.build_tree(self.train_data, self.features)

    def predict(self, test_data):
        result = test_data.apply(lambda row: self.run_tree(row, self.root), axis=1)

        return result
