from flask import Flask, request, render_template
from pycaret.classification import load_model, predict_model
import pandas as pd

app = Flask(__name__)

# Load the model and pipeline
model = load_model('mushroom')  # Load the model without the .pkl extension

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form and convert it into a DataFrame
    form_data = {key: [value] for key, value in request.form.items()}
    input_data = pd.DataFrame.from_dict(form_data)
    
    # Predict using the loaded model
    predictions = predict_model(model, data=input_data)
    
    # Extract the predicted label
    output = predictions['prediction_label'][0]
    
    return render_template('index3.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)