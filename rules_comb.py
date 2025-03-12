def evaluate_combined_rules_by_index(X, y, rule_indices, rule_mapping):
    """
    Evaluate combined performance of multiple rules selected by their numeric indices.

    Parameters:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Binary target variable.
        rule_indices (list of int): List of numeric indices of rules to evaluate.
        rule_mapping (dict): Dictionary mapping numeric indices to rule strings.

    Returns:
        dict: Dictionary containing Accuracy, Precision, Recall, F1, Coverage.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    selected_rules = [rule_mapping[idx] for idx in rule_indices]

    combined_rule = ' | '.join(f'({rule})' for rule in selected_rules)

    condition = X.eval(combined_rule)
    coverage = condition.mean()

    if coverage == 0:
        print("Combined rule coverage is zero. Check selected rules for correctness.")
        return None

    y_pred = pd.Series(0, index=y.index)
    y_pred[condition] = 1

    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'Coverage': coverage
    }

    return metrics
