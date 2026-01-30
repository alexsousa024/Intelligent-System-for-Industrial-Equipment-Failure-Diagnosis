'''
Bayesian Network Model Evaluation, including:
    - Predict target variable for input data
    - Evaluate model performance with accuracy, confusion matrix, classification report
'''

import numpy as np
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def predict_row_proba(row, infer, features, target, positive_state):
    evidence = {f: row[f] for f in features}
    q = infer.query([target], evidence=evidence)

    p1 = q.values[positive_state]
    return float(p1)

def predict_df(df, model, features, target, threshold=0.5, positive_state=1):
    infer = VariableElimination(model)
    
    y_prob = []
    y_pred = []
    
    for _, row in df.iterrows():
        p = predict_row_proba(row, infer, features, target, positive_state)
        y_prob.append(p)
        y_pred.append(int(p >= threshold))
        
    return np.array(y_pred), np.array(y_prob)

def evaluate_model(model, df_test, features, target, threshold=0.5, positive_state=1, positive_label="Yes", negative_label="No"):
    y_true = df_test[target].values
    y_pred, y_prob = predict_df(df_test, model, features, target, threshold, positive_state)
    y_pred_labels = np.where( y_pred==1, positive_label, negative_label)

    acc = accuracy_score(y_true, y_pred_labels)
    print("\nAccuracy:", acc)
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred_labels))
    print("\nClassification Report:\n", classification_report(y_true, y_pred_labels))
    return {
        "accuracy": acc,
        "y_true": y_true,
        "y_pred": y_pred_labels,
        "y_prob": y_prob
    }
