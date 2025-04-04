# Rules_Engine

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


    What This System Does

NarratML helps translate "why" a machine learning model made a prediction into simple human language.

Instead of just saying:

“This customer is likely to churn.”
It can say something like:

“The customer’s high usage of the word ‘cancel,’ their low income, and their rural region contributed to this prediction.”
This helps people understand, trust, and even act on predictions—especially in fields like healthcare, finance, education, or customer support.

How It Works (Simplified)

Think of it like a 5-step detective process:

1. Look at the Data Used in the Prediction
Your system starts with structured data (basically, a table). It looks at:

Whether certain words show up (e.g., “refund”, “cancel”)
What region or category something belongs to
Numeric values like income or age
Each column in the table represents a type of information (called a feature).

2. Use SHAP to See What Influenced the Prediction
SHAP (SHapley Additive exPlanations) is like a calculator that measures how much each feature (like income or word usage) pushed the model’s prediction up or down.

It tells us:

Which pieces of info mattered the most
Whether each one helped increase or decrease the prediction
3. Translate the Raw Math into Human-Like Rules
Now comes the fun part. Instead of saying:

“Income SHAP = +0.23”
The system translates it to something like:

“Higher income increased the likelihood of this result.”
It has different rules for:

Words: “The presence of the word ‘cancel’ increased the prediction.”
Categories: “Customers in this region tend to have a higher risk.”
Numbers: “Higher values for time spent on hold are linked to higher churn.”
4. Add Context from Partial Dependence (for Numbers)
For numeric values (like age, balance, time), the system can even analyze patterns—like:

“As age increases, the risk goes down.”
“People with income over $70k are less likely to leave.”
This step gives general insights that help avoid false conclusions.

5. Send Everything to a Generative AI Model
The final step is where magic happens.

The system takes the top 3–5 most important reasons (in sentence form), sends them to an AI (like AWS Bedrock’s models), and asks:

“Please summarize these into 1-2 simple sentences.”
And you get back:

“This prediction was driven by the presence of certain risk-indicating words, a high time on hold, and location data associated with churn.”
This final summary is what the end user sees—clean, readable, trustworthy.

Why This Matters

Transparency: People know why a prediction was made.
Trust: Businesses and users are more likely to accept the model.
Actionable Insight: Teams can fix or act on the causes of risky predictions.
Core Components

Here’s a non-technical breakdown of the building blocks:

Component	Purpose
Feature Profiler	Identifies what kind of data each feature is (word, number, category)
SHAP Processor	Figures out how important each feature is for a specific prediction
Rule Generator	Converts that info into clear text for each feature
PDP Analyzer	Adds general context for number-based features (e.g., “higher is better”)
LLM (Bedrock)	Takes all the pieces and summarizes it into human-style sentences
NarratML Explainer	The central brain that coordinates the entire process
Example Explanation

Prediction: Customer likely to cancel service
NarratML Output:
“This prediction was influenced by the presence of the word 'cancel' in recent messages, a low income level, and a region typically associated with higher churn.”
Would you like a version of this in presentation or PDF format too? I can create that for internal/executive audiences.
