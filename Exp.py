import os
import yaml

def create_structure(structure, root="."):
    for name, content in structure.items():
        dir_path = os.path.join(root, name)
        os.makedirs(dir_path, exist_ok=True)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    open(os.path.join(dir_path, item), "w").close()
                elif isinstance(item, dict):
                    create_structure(item, root=dir_path)

if __name__ == "__main__":
    with open("project_structure.yml", "r") as f:
        structure = yaml.safe_load(f)
    create_structure(structure)
    print("Project scaffold created successfully.")


class FeatureProfiler:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.metadata = {}

    def infer_feature_types(self):
        # Detect and label feature types
        # e.g. {"income": "numerical", "region_encoded": "mean_encoded"}
        pass

    def get_metadata(self):
        return self.metadata


step 1

class PredictionExplainer:
    def __init__(self, model, shap_explainer, feature_metadata, pdp_data, llm_client):
        self.model = model
        self.shap_explainer = shap_explainer
        self.feature_metadata = feature_metadata
        self.pdp_data = pdp_data
        self.llm_client = llm_client

    def explain_instance(self, input_data: dict, top_n_features=5) -> dict:
        # Step 1: Get SHAP values for instance
        # Step 2: Rank and interpret features based on type
        # Step 3: Generate structured rule text
        # Step 4: Pass to LLM with prompt
        # Step 5: Return explanation + trace
        pass

3. 

def get_shap_values_for_instance(shap_explainer, input_instance):
    # Return dictionary of SHAP values for all features
    pass

def rank_top_features(shap_values, n=5):
    # Sort and return top N influential features
    pass

def generate_rule(feature_name, value, shap_value, feature_type, metadata, pdp_data=None):
    # Apply logic based on feature type to produce human-readable explanation
    pass


class BedrockLLM:
    def __init__(self, model_id, region="us-east-1"):
        # Setup boto3 Bedrock client
        pass

    def generate_explanation(self, feature_insights):
        # Construct prompt using insights
        # Return LLM response
        pass
1.Shap Processor

# shap_processor.py
import numpy as np

# shap_processor.py
import shap
import pandas as pd

def get_shap_explainer(model, X):
    """
    Returns a SHAP TreeExplainer for an XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    return explainer

def get_shap_values(model, X):
    """
    Computes SHAP values for all rows in X using a TreeExplainer.
    
    Returns:
        shap_df (pd.DataFrame): SHAP values, same shape as X
        explainer (shap.Explainer): SHAP explainer for reuse
    """
    explainer = get_shap_explainer(model, X)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    return shap_df, explainer



2. explainer.py


# explainer.py
from narratml.shap_processor import get_shap_values_for_instance, rank_top_features
from narratml.rule_generator import generate_rule
from narratml.llm_interface import BedrockLLM  # Placeholder for now

class PredictionExplainer:
    def __init__(self, model, shap_explainer, feature_metadata, pdp_data=None, llm_client=None):
        self.model = model
        self.shap_explainer = shap_explainer
        self.feature_metadata = feature_metadata
        self.pdp_data = pdp_data or {}
        self.llm_client = llm_client

    def explain_instance(self, input_data: dict, top_n_features=5) -> dict:
        shap_dict = get_shap_values_for_instance(self.shap_explainer, [input_data])
        top_features = rank_top_features(shap_dict, top_n_features)

        feature_insights = []
        for feature, shap_value in top_features:
            feature_type = self.feature_metadata.get(feature, "numerical")
            value = input_data[feature]
            pdp = self.pdp_data.get(feature)
            rule = generate_rule(feature, value, shap_value, feature_type, pdp)
            feature_insights.append(rule)

        # Send to LLM
        if self.llm_client:
            summary = self.llm_client.generate_explanation(feature_insights)
        else:
            summary = " | ".join(feature_insights)

        return {
            "summary": summary,
            "feature_insights": feature_insights,
            "top_features": top_features
        }

2. rule generator 


# rule_generator.py

def generate_rule(feature_name, value, shap_value, feature_type, pdp=None):
    direction = "positively" if shap_value > 0 else "negatively"

    if feature_type == "word_indicator":
        if value == 1:
            return f"The presence of the word '{feature_name}' {direction} influenced the prediction."
        else:
            return f"The absence of the word '{feature_name}' {direction} influenced the prediction."

    elif feature_type == "mean_encoded":
        return (
            f"The category '{value}' for '{feature_name}' is associated with a "
            f"{direction} impact based on target averages."

        )

    elif feature_type == "numerical":
        rule = f"The value {value} for '{feature_name}' {direction} contributed to the prediction."
        if pdp and hasattr(pdp, "trend"):
            if pdp["trend"] == "increasing":
                rule += " Higher values are generally more predictive."
            elif pdp["trend"] == "decreasing":
                rule += " Lower values are generally more predictive."
        return rule

    elif feature_type == "tfidf":
        return f"The term '{feature_name}' with weight {value:.2f} {direction} influenced the prediction."

    else:
        return f"The feature '{feature_name}' with value {value} {direction} influenced the prediction."


interface 

# llm_interface.py
import boto3
import json

class BedrockLLM:
    def __init__(self, model_id, region="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def generate_explanation(self, feature_insights):
        prompt = self._build_prompt(feature_insights)

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"prompt": prompt, "max_tokens": 200}),
            contentType="application/json",
            accept="application/json"
        )

        output = json.loads(response['body'].read())
        return output['completion']

    def _build_prompt(self, insights):
        text = "\n".join(f"- {insight}" for insight in insights)
        return (
            "You are an AI assistant that explains machine learning predictions in plain English.\n"
            "Given the following feature-level insights:\n\n"
            f"{text}\n\n"
            "Generate a short, non-technical explanation (1–2 sentences) of why this prediction was made."
        )


# feature_profiles.py

# feature_profiles.py
import numpy as np
import pandas as pd
import warnings

class FeatureProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metadata = {}

    def infer_feature_types(self):
        for col in self.df.columns:
            series = self.df[col]
            unique_vals = series.dropna().unique()
            nunique = len(unique_vals)

            if nunique == 2:
                # Check if it looks like a proper indicator
                if set(unique_vals) == {0, 1}:
                    self.metadata[col] = "indicator"
                else:
                    self.metadata[col] = "indicator"
                    warnings.warn(
                        f"⚠️ Column '{col}' has 2 unique values but is not [0, 1]: {unique_vals}"
                    )

            elif pd.api.types.is_numeric_dtype(series):
                # Check if it's an encoding
                value_counts = series.value_counts(normalize=True)
                if all(value_counts < 0.95) and any(value_counts >= 0.05):
                    self.metadata[col] = "encoding"
                else:
                    self.metadata[col] = "numerical"

            else:
                self.metadata[col] = "unknown"

        return self.metadata

    def get_metadata(self):
        return self.metadata



# pdp_analyzer.py
import numpy as np

def analyze_pdp_trend(x, y, threshold=0.1):
    """
    Analyzes trend in PDP line (x = values, y = PDP output)
    Returns: "increasing", "decreasing", "flat", or "non-monotonic"
    """
    slope = np.polyfit(x, y, 1)[0]

    if slope > threshold:
        return "increasing"
    elif slope < -threshold:
        return "decreasing"
    elif abs(slope) <= threshold:
        return "flat"
    else:
        return "non-monotonic"

def create_pdp_profile(feature_name, x, y):
    trend = analyze_pdp_trend(x, y)
    return {
        "feature": feature_name,
        "trend": trend,
        "x": x,
        "y": y
    }

# utils.py

def pretty_print_dict(d):
    for k, v in d.items():
        print(f"{k}: {v}")

def sort_dict_by_abs_value(d, reverse=True):
    return dict(sorted(d.items(), key=lambda item: abs(item[1]), reverse=reverse))

# narratml/__init__.py

from .explainer import PredictionExplainer
from .feature_profiles import FeatureProfiler
from .shap_processor import get_shap_values_for_instance, rank_top_features
from .rule_generator import generate_rule
from .llm_interface import BedrockLLM
from .pdp_analyzer import analyze_pdp_trend, create_pdp_profile




