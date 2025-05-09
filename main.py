import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data.csv")

dataset.drop(columns=['customerID'], inplace=True)

encoder = LabelEncoder()
for column in dataset.select_dtypes(include=['object']).columns:
    dataset[column] = encoder.fit_transform(dataset[column])