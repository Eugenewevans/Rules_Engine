import pandas as pd
from itertools import product

def evaluate_rules(df, features, target):
    results = []

    # Generate all combinations of categorical features
    feature_combinations = list(product(*[df[col].dropna().unique() for col in features]))

    # Evaluate each rule combination
    for idx, values in enumerate(feature_combinations):
        condition = pd.Series([True] * len(df))

        # Building rule conditions
        for feature, value in zip(features, values):
            condition &= (df[feature] == value)

        # Calculate confusion matrix components
        tp = df[condition & (df[target] == 1)].shape[0]
        fp = df[condition & (df[target] == 0)].shape[0]
        fn = df[~condition & (df[target] == 1)].shape[0]
        
        # Calculate performance metrics
        coverage = condition.mean()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        results.append({
            'Rule ID': idx,
            'Rule': ", ".join(f"{feature}={value}" for feature, value in zip(features, values)),
            'Coverage': coverage,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

    rules_df = pd.DataFrame(results)
    rules_df.sort_values(['F1 Score', 'Precision', 'Coverage'], ascending=[False, False, False], inplace=True)
    rules_df.reset_index(drop=True, inplace=True)
    
    return rules_df


def select_rules(rules_df, selected_rule_ids):
    return rules_df[rules_df['Rule ID'].isin(selected_rule_ids)].reset_index(drop=True)
