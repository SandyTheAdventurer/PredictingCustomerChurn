import gradio as gr
import main
import numpy as np
import pandas as pd
from main import clf_rf, clf_log, accuracy_score_rf, accuracy_score_lr, brier_score_rf, brier_score_lr, roc_rf, roc_lr

def eda(Graphs):
    match Graphs:
        case "Customer Churn":
            return gr.Image("graphs/EDAGraphs/Churn.png")
        case "Contract":
            return gr.Image("graphs/EDAGraphs/Contract.png")
        case "Dependents":
            return gr.Image("graphs/EDAGraphs/Dependents.png")
        case "Device Protection":
            return gr.Image("graphs/EDAGraphs/DeviceProtection.png")
        case "Heatmap":
            return gr.Image("graphs/EDAGraphs/Heatmap.png")
        case "Monthly Charges":
            return gr.Image("graphs/EDAGraphs/MonthlyCharges.png")
        case "Online Backup":
            return gr.Image("graphs/EDAGraphs/OnlineBackup.png")
        case "Online Security":
            return gr.Image("graphs/EDAGraphs/OnlineSecurity.png")
        case "Paperless Billing":
            return gr.Image("graphs/EDAGraphs/PaperlessBilling.png")
        case "Partner":
            return gr.Image("graphs/EDAGraphs/Partner.png")
        case "Payment Method":
            return gr.Image("graphs/EDAGraphs/PaymentMethod.png")
        case "Senior Citizen":
            return gr.Image("graphs/EDAGraphs/SeniorCitizen.png")
        case "Tech Support":
            return gr.Image("graphs/EDAGraphs/TechSupport.png")
        case "Tenure":
            return gr.Image("graphs/EDAGraphs/tenure.png")

def result(Graphs):
    match Graphs:
        case "Correlation":
            inf = """
Inferences:

    - Churn is perfectly correlated with itself (1.0).

    - Tenure and Contract have strong negative correlation with Churn.

    - MonthlyCharges and PaperlessBilling have moderate positive correlation with Churn.

    - Gender, PhoneService, and MultipleLines have near-zero correlation.

    - OnlineSecurity, TechSupport, and DeviceProtection have moderate negative correlation.
"""
            return [gr.Image("graphs/EDAGraphs/Correlation.png"), inf]
        case "Confusion Matrix for Random Forest":
            inf = """
Inferences:

    - True Positives (181) are lower than Logistic Regression’s (215), indicating worse recall for churn.

    - False Negatives (192) are higher – more churners are missed.

    - False Positives (104) are slightly higher than logistic regression’s — marginally more false alarms.

    - True Negatives (932) are comparable – performs similarly for non-churn cases.

    - Overall: Slightly poorer at identifying churn compared to Logistic Regression in this case.

"""
            return [gr.Image("graphs/OutputGraphs/ConfusionMatrixRandomForest.png"), inf]
        case "Confusion Matrix for Logistic regression":
            inf = """
Inferences:

    - True Negatives (936) and True Positives (215) indicate good performance in correctly identifying both classes.

    - False Negatives (158): A moderate number of actual churns are missed, which could be critical in business decisions.

    - False Positives (100): Some non-churning customers are predicted to churn — might lead to unnecessary retention efforts.

    - Model Bias: Slight bias towards predicting the majority class (non-churn).

    - Overall: Reasonable balance, but recall for churn could be improved.

"""
            return [gr.Image("graphs/OutputGraphs/ConfusionMatrixLogistic.png"), inf]
        case "SHAP analysis for Random Forest":
            inf = """
Inferences:

    - Bimodal Distribution: Indicates two clear groups — likely senior citizens and non-seniors with distinct behavior patterns.

    - Near-Zero SHAP values: SeniorCitizen has limited standalone predictive power.

    - Interaction: The variable may influence predictions when combined with other features (e.g., internet service or contract).

    - Red vs Blue Dots: Represents SHAP values across classes — they’re symmetric, confirming weak influence.

    - Overall: SeniorCitizen is not a key predictor by itself in this model.
"""
            return [gr.Image("graphs/OutputGraphs/SHAP_RandomForest_Summary.png"), inf]
        case "SHAP analysis for Logistic Regression":
            inf = """
Inferences:

    - Top Feature: tenure is the most influential in predicting churn — lower tenure likely increases churn risk.

    - MonthlyCharges & Contract also have strong effects — customers on monthly or expensive plans may churn more.

    - Security-related services (e.g., OnlineSecurity, TechSupport) have moderate influence — presence may reduce churn.
    
    - PaperlessBilling and OnlineBackup show notable contributions, possibly associated with digital-savvy customers.

    - Features like DeviceProtection, Partner, and PaymentMethod have minimal impact.
"""
            return [gr.Image("graphs/OutputGraphs/SHAP_Logistic_Summary.png"), inf]

def metrics(Algorithms):
    match Algorithms:
        case "Random Forest":
            df = pd.DataFrame(clf_rf)
            df = df.drop(columns = ['accuracy', 'macro avg', 'weighted avg']).T
            df = df.reset_index().rename(columns={'index': 'class'})
            df_clf = gr.DataFrame(
                value = df
            )
            df_acc = gr.DataFrame(
                headers = ['Accuracy Score', 'Brier Score', 'ROC Score'],
                value = [list([accuracy_score_rf, brier_score_rf, roc_rf])],
            )
            return df_clf, df_acc
        
        case "Logistic Regression":
            df = pd.DataFrame(clf_log)
            df = df.drop(columns = ['accuracy', 'macro avg', 'weighted avg']).T
            df = df.reset_index().rename(columns={'index': 'class'})
            df_clf = gr.DataFrame(
                value = df
            )
            df_acc = gr.DataFrame(
                headers = ['Accuracy Score', 'Brier Score', 'ROC Score'],
                value = [list([accuracy_score_lr, brier_score_lr, roc_lr])],
            )
            return df_clf, df_acc

with gr.Blocks() as Output:
    gr.Markdown("View Exploratory data Analysis and Output")
    with gr.Tab("EDA Graphs"):
        eda_input = gr.Radio(["Customer Churn", "Contract", "Dependents", "Device Protection", "Heatmap", "Monthly Charges", "Online Backup", "Online Security", "Paperless Billing", "Partner", "Payment Method", "Senior Citizen", "Tech Support", "Tenure"], show_label= False)
        eda_output = gr.Image()

        eda_input.change(fn = eda, inputs= eda_input, outputs= eda_output)

    with gr.Tab("Output Graphs"):
        result_input = gr.Radio(["Correlation", "Confusion Matrix for Random Forest", "Confusion Matrix for Logistic regression", "SHAP analysis for Random Forest", "SHAP analysis for Logistic Regression"], show_label = False)

        result_output = [gr.Image(), gr.Markdown()]

        result_input.change(fn = result, inputs=result_input, outputs = result_output)

    with gr.Tab("Performance Metrics"):
        algorithm = gr.Radio(["Random Forest", "Logistic Regression"], show_label= False)

        metrics_output = [gr.DataFrame(), gr.DataFrame()]

        algorithm.change(fn = metrics, inputs = algorithm, outputs = metrics_output)


Output.launch()