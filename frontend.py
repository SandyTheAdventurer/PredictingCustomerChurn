import gradio as gr
import main
import numpy as np
import pandas as pd
from main import clf_rf, clf_log, accuracy_score_rf, accuracy_score_lr, breier_score_rf, breier_score_lr, roc_rf, roc_lr

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
            inf = "Enter inference here"
            return [gr.Image("graphs/EDAGraphs/Correlation.png"), inf]
        case "Confusion Matrix for Random Forest":
            inf = "Enter inference here"
            return [gr.Image("graphs/OutputGraphs/ConfusionMatrixRandomForest.png"), inf]
        case "Confusion Matrix for Logistic regression":
            inf = "Enter inference here"
            return [gr.Image("graphs/OutputGraphs/ConfusionMatrixLogistic.png"), inf]
        case "SHAP analysis for Random Forest":
            inf = "Enter inference here"
            return [gr.Image("graphs/OutputGraphs/SHAP_RandomForest_Summary.png"), inf]
        case "SHAP analysis for Logistic Regression":
            inf = "Enter inference here"
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
                headers = ['Accuracy Score', 'Breier Score', 'ROC Score'],
                value = [list([accuracy_score_rf, breier_score_rf, roc_rf])],
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
                headers = ['Accuracy Score', 'Breier Score', 'ROC Score'],
                value = [list([accuracy_score_lr, breier_score_lr, roc_lr])],
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

        result_output = [gr.Image(), gr.Label()]

        result_input.change(fn = result, inputs=result_input, outputs = result_output)

    with gr.Tab("Performance Metrics"):
        algorithm = gr.Radio(["Random Forest", "Logistic Regression"], show_label= False)

        metrics_output = [gr.DataFrame(), gr.DataFrame()]

        algorithm.change(fn = metrics, inputs = algorithm, outputs = metrics_output)


Output.launch()