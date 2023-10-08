import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Check if the pre-trained model exists, and load it if it does
model_file_path = 'fuel_efficiency_model.pkl'
if os.path.exists(model_file_path):
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
else:
    # Load the dataset from the CSV file (replace with your actual file path)
    data = pd.read_csv('fuel_efficiency_data.csv')

    # Define your independent variables (X) and dependent variable (y)
    X = data[['Engine_Size', 'Weight', 'Aerodynamics']]
    y = data['Fuel_Efficiency']

    # Split the dataset into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model (you can use different evaluation metrics)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training R-squared score: {train_score}")
    print(f"Testing R-squared score: {test_score}")

    # Save the trained model to a file
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_fuel_efficiency():
    # Get user input
    engine_size = float(request.form['engine_size'])
    weight = float(request.form['weight'])
    aerodynamics = float(request.form['aerodynamics'])

    # Make a prediction using the model
    input_data = pd.DataFrame({
        'Engine_Size': [engine_size],
        'Weight': [weight],
        'Aerodynamics': [aerodynamics]
    })
    predicted_efficiency = model.predict(input_data)[0]

    return render_template('result.html', predicted_efficiency=predicted_efficiency)


if __name__ == '__main__':
    app.run(debug=True)
