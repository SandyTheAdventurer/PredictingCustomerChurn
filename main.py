import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay, brier_score_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import shap
import warnings

warnings.filterwarnings("ignore")

shap.initjs()

dataset = pd.read_csv("data.csv")

# Dropping customerID which is unique for each customer
# and does not provide any useful information for prediction
dataset.drop(columns=['customerID'], inplace=True)

# Encoding categorical variables and Scaling numerical variables
encoder = LabelEncoder()
scaler = StandardScaler()
for column in dataset.select_dtypes(include=['int64', 'float64']).columns:
    dataset[column] = scaler.fit_transform(dataset[column].values.reshape(-1, 1))
for column in dataset.select_dtypes(include=['object']).columns:
    dataset[column] = encoder.fit_transform(dataset[column])

# Plotting the correlation to find the most important features
fig, ax = plt.subplots(figsize=(16, 10))
corr=dataset.corr()["Churn"]
ax.set_xticklabels(corr.index, rotation=45, ha='right', fontsize=10)

sns.barplot(x=corr.index, y=corr.values, ax=ax)
plt.savefig("graphs/EDAGraphs/Correlation.png")

# Based on the correlation plot, the following features are removed
dataset.drop(columns=['gender', 'PhoneService', 'MultipleLines', 'InternetService', 'StreamingTV', 'StreamingMovies', 'TotalCharges'], inplace=True)
# Multivariate analysis
fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
plt.savefig("graphs/EDAGraphs/Heatmap.png")

for column in dataset.columns:
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.kdeplot(dataset[column],ax=ax)
    plt.savefig(f"graphs/EDAGraphs/{column}.png")

# Splitting the dataset into training and testing sets
y = dataset.pop("Churn").values
X = dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model

rf_clf = RandomForestClassifier(n_estimators = 100) 
rf_clf.fit(X_train, y_train)
y_rf_pred = rf_clf.predict(X_test)

# Evaluating Random Forest 
print("\n\t\t\tRandom Forest Classifier:\n")
print("Classification Report for Random Forest:")
clf_rf = classification_report(y_test, y_rf_pred, output_dict= True)
print(classification_report(y_test, y_rf_pred))

cm = confusion_matrix(y_test, y_rf_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_clf.classes_)
display.plot(cmap=plt.cm.Blues)
plt.savefig("graphs/OutputGraphs/ConfusionMatrixRandomForest.png")

breier_score_rf = brier_score_loss(y_test, y_rf_pred)
accuracy_score_rf = accuracy_score(y_test, y_rf_pred)
roc_rf = roc_auc_score(y_test, y_rf_pred)

print("Brier Score Loss:")
print(breier_score_rf)
print("Accuracy Score:")
print(accuracy_score_rf)
print("ROC AUC Score:")
print(roc_rf)


explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("graphs/OutputGraphs/SHAP_RandomForest_Summary.png")


# Logistic Regression Model
logistic = LogisticRegression()

logistic.fit(X_train, y_train)

y_log_pred = logistic.predict(X_test)

# Evaluating logistic
print("\t\t\tLogistic Regression:\n")
print("Classification Report for Logistic Regression:")
clf_log = classification_report(y_test, y_log_pred, output_dict= True)
print(classification_report(y_test, y_log_pred))

cm = confusion_matrix(y_test, y_log_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic.classes_)
display.plot(cmap=plt.cm.Blues)
plt.savefig("graphs/OutputGraphs/ConfusionMatrixLogistic.png")

breier_score_lr = brier_score_loss(y_test, y_log_pred)
accuracy_score_lr = accuracy_score(y_test, y_log_pred)
roc_lr = roc_auc_score(y_test, y_log_pred)

print("Brier Score Loss:")
print(breier_score_lr)
print("Accuracy Score:")
print(accuracy_score_lr)
print("ROC AUC Score:")
print(roc_lr)

# SHAP Analysis for logistic regression
explainer = shap.LinearExplainer(logistic, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("graphs/OutputGraphs/SHAP_Logistic_Summary.png")