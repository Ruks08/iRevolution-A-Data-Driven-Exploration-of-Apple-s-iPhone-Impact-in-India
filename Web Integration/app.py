
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join('model', 'model.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "\n❌ model.pkl not found!\n"
        "Please run: python model/train_model.py\n"
    )

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data['mrp']),
            float(data['discount_percentage']),
            float(data['number_of_ratings']),
            float(data['number_of_reviews']),
            float(data['star_rating']),
            float(data['ram_gb']),
            float(data['storage_gb'])
        ]])

        predicted_price = model.predict(features)[0]

        low  = round(predicted_price * 0.95 / 100) * 100
        high = round(predicted_price * 1.05 / 100) * 100

        return jsonify({
            'success': True,
            'predicted_price': int(predicted_price),
            'price_low':  int(low),
            'price_high': int(high)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
