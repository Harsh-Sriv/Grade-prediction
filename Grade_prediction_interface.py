import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load the trained model and scaler
model = joblib.load('linear_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Function to get user input for each feature
def get_user_input(feature_names):
    user_input = {feature: 0 for feature in feature_names}
    
    # Continuous and ordinal features
    continuous_ordinal = ['age', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    for feature in continuous_ordinal:
        if feature in ['freetime', 'goout', 'Dalc', 'Walc', 'health']:
            print(f"{feature} (1-very low to 5-very high):")
        user_input[feature] = float(input(f"Enter value for {feature}: "))
    
    # School
    school = input("Enter school (GP or MS): ").upper()
    user_input[school] = 1
    
    # Sex
    sex = input("Enter sex (F or M): ").upper()
    user_input[sex] = 1
    
    # Address
    address = input("Enter address type (U for Urban, R for Rural): ").upper()
    user_input[address] = 1
    
    # Travel time
    print("Travel time to school:")
    print("1: <15 min, 2: 15-30 min, 3: 30 min-1 hour, 4: >1 hour")
    travel_time = int(input("Enter travel time category (1-4): "))
    user_input[f'(T){["<15", "15-30", "30-1", ">1"][travel_time-1]}'] = 1
    user_input['T'] = 1  # Set 'T' to 1 as it's always present when travel time is specified
    
    # Study time
    print("Weekly study time:")
    print("1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours")
    study_time = int(input("Enter study time category (1-4): "))
    user_input[f'(S){["<2", "2-5", "5-10", ">10"][study_time-1]}'] = 1
    
    # Failures
    failures = int(input("Enter number of past class failures (0-3): "))
    user_input[f'F{failures}'] = 1
    
    # Family relationship quality
    print("Quality of family relationships:")
    print("1: very bad, 2: bad, 3: average, 4: good, 5: excellent")
    famrel = int(input("Enter family relationship quality (1-5): "))
    user_input[['very bad', 'Bad', 'Average', 'Good', 'Excellent'][famrel-1]] = 1
    
    # Binary features
    binary_features = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for feature in binary_features:
        value = input(f"Does {feature} apply? (yes/no): ").lower() == 'yes'
        user_input[feature] = int(value)
    
    return user_input

# Main prediction function
def predict_grade():
    # Get the feature names (excluding 'G3' which is the target)
    feature_names = ['age', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 
                     'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'GP', 'MS', 'F', 'M', 
                     'A', 'T', '(T)<15', '(T)15-30', '(T)30-1', '(T)>1', '(S)<2', '(S)2-5', '(S)5-10', '(S)>10', 
                     'F0', 'F1', 'F2', 'F3', 'very bad', 'Bad', 'Average', 'Good', 'Excellent']

    # Get user input
    user_input = get_user_input(feature_names)

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure the order of columns matches the order used during training
    input_df = input_df[feature_names]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    print(f"Predicted grade (G3): {prediction[0]:.2f}")

# Run the prediction
if __name__ == "__main__":
    print("Welcome to the Student Grade Prediction System")
    while True:
        predict_grade()
        if input("Do you want to make another prediction? (yes/no): ").lower() != 'yes':
            break
    print("Thank you for using the Student Grade Prediction System!")