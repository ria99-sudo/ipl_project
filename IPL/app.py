from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle  # If you need to load scalers/encoders

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = keras.models.load_model("lstm1.keras")

# Load encoders/scalers if needed
encoder = pickle.load(open("encoder.pkl", "rb"))
scalerx = pickle.load(open("scalerx.pkl", "rb"))
scalery = pickle.load(open("scalery.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.json
    team1 = data['team1']
    team2 = data['team2']
    venue = data['venue']
    over = data['over']
    ball = data['ball']

    # Preprocess input
    input_df = pd.DataFrame([[team1, team2, venue]], columns=['batting_team', 'bowling_team', 'venue'])
    encoded_input = encoder.transform(input_df)
    extrainput = np.array([[over, ball]])
    scaled_extra = scalerx.transform(extrainput)

    X_input = np.hstack((encoded_input, scaled_extra))
    X_input = X_input.reshape(1, 1, -1)

    # Make prediction
    prediction = model.predict(X_input)
    prediction = scalery.inverse_transform(prediction)

    # Send response
    return jsonify({
        "predicted_runs": round(prediction[0][0]),
        "predicted_wickets": round(prediction[0][1])
    })

if __name__ == '__main__':
    app.run(debug=True)
