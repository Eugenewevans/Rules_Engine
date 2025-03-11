import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to recursively extract decision rules from a trained decision tree
def extract_rules(tree, feature_names):
    tree_ = tree.tree_

    # Inner recursive function
    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Extract the feature name and threshold for current node
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Define conditions for left (<=) and right (>) branches
            left_conditions = conditions + [f'({name} <= {threshold:.4f})']
            right_conditions = conditions + [f'({name} > {threshold:.4f})']

            # Recursively generate rules for child nodes
            yield from recurse(tree_.children_left[node], left_conditions)
            yield from recurse(tree_.children_right[node], right_conditions)
        else:
            # Leaf node: return accumulated conditions
            yield ' and '.join(conditions), tree_.value[node]

    return list(recurse(0, []))


# Evaluate a given rule on dataset and compute performance metrics
def evaluate_rule(X, y, rule):
    # Evaluate the rule condition
    condition = X.eval(rule)
    coverage = condition.mean()

    if coverage == 0:
        return None

    # Create predictions based on the rule
    y_pred = pd.Series(0, index=y.index)
    y_pred[condition] = 1

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'Coverage': coverage
    }

    return metrics


# Generate decision rules based on a trained decision tree classifier
def generate_decision_rules(X, y, metric='Accuracy', max_depth=4, max_rules=10, min_coverage=0.01):
    # Train decision tree classifier with provided constraints
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=int(len(X) * min_coverage))
    clf.fit(X, y)

    # Extract rules from trained classifier
    raw_rules = extract_rules(clf, X.columns)

    rule_evaluations = []
    for rule, _ in raw_rules:
        metrics = evaluate_rule(X, y, rule)
        if metrics and metrics['Coverage'] >= min_coverage:
            rule_evaluations.append({'Rule': rule, **metrics})

    # Create DataFrame, sort by chosen metric, and limit to max_rules
    rules_df = pd.DataFrame(rule_evaluations)
    rules_df.sort_values(by=metric, ascending=False, inplace=True)

    return rules_df.head(max_rules)


# Example usage demonstrating how to setup and execute the code
if __name__ == '__main__':
    # Example: Generating synthetic classification dataset
    from sklearn.datasets import make_classification

    # Generate synthetic data (adjust n_samples and n_features as needed)
    X_sample, y_sample = make_classification(n_samples=50000, n_features=5, random_state=42)

    # Convert to pandas DataFrame for easy evaluation
    X_df = pd.DataFrame(X_sample, columns=[f'feature_{i}' for i in range(1, 6)])
    y_series = pd.Series(y_sample)

    # Generate rules with specific settings
    rules = generate_decision_rules(
        X=X_df,                     # Feature DataFrame
        y=y_series,                 # Target variable
        metric='Accuracy',          # Metric to rank rules ('Accuracy', 'Precision', 'Recall', 'F1')
        max_depth=4,                # Maximum depth of tree (complexity)
        max_rules=5,                # Maximum number of rules to return
        min_coverage=0.02           # Minimum coverage threshold for a rule
    )

    # Output generated rules and their metrics
    print(rules)
