from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

def fet_sel(data):
    fsel_loaded = joblib.load('trained_feature_selector.joblib')

    columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
        'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
        'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
        'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
        'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
        'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
        'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
        'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
        'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
        'Soil_Type39', 'Soil_Type40']
    
    x_numpy = np.array(data)
    x_reshape = x_numpy.reshape(1, 54)
    df = pd.DataFrame(x_reshape,columns=columns)
    df_fet_sel = fsel_loaded.transform(df)

    return df_fet_sel


# Initialize Flask application
app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route('/')
def home():
    return """Welcome to the prediction API!
              Use the /predict endpoint to get predictions."""

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    X_test = fet_sel(features)
    prediction = model.predict(X_test)
    response = {
        'prediction': int(prediction[0]+1)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)