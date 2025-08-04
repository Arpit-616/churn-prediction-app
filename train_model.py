import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

def load_and_preprocess_data():
    """Load and preprocess the Telco customer churn dataset"""
    # Load the dataset
    df = pd.read_csv('../p1/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print(f"Dataset loaded with {len(df)} records and {len(df.columns)} columns")
    print(f"Churn rate: {df['Churn'].value_counts(normalize=True)}")
    
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # Convert SeniorCitizen to int if it's not already
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # Convert Churn to binary
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    # Select features for the model
    feature_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X = df[feature_columns].copy()
    y = df['Churn']
    
    # Handle categorical variables
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Create dummy variables
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    return X_encoded, y

def train_model(X, y):
    """Train the Random Forest model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print model performance
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return rf_model, scaler, X.columns.tolist()

def save_model(model, scaler, feature_columns):
    """Save the trained model and scaler"""
    # Save the model
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns for reference
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("\nModel and scaler saved successfully!")
    print(f"Model saved as: churn_model.pkl")
    print(f"Scaler saved as: scaler.pkl")
    print(f"Feature columns saved as: feature_columns.pkl")

def main():
    """Main function to train the model"""
    print("Starting Customer Churn Model Training...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features
    X, y = prepare_features(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Train model
    model, scaler, feature_columns = train_model(X, y)
    
    # Save model
    save_model(model, scaler, feature_columns)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 