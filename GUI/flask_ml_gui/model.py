# model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('DARWIN.csv')
df.drop(columns=['ID'], inplace=True)

# Prepare features and target variable
X = df.drop(columns=['class'])
y = df['class']

# Split the data into training and testing sets to ensure the model is validated properly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the feature names used during training
feature_names = X.columns.tolist()

# Train the model
def train_model():
    best_iter = 1000  # Increased number of iterations
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=best_iter))
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Model and feature names trained and saved.")

# Predict function
def predict(input_data):
    pipeline = joblib.load('model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = pipeline.predict(input_df)
    return prediction[0]

# Initial model training (Run this once to train and save the model)
if __name__ == "__main__":
    train_model()
