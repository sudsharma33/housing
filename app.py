from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

data = pd.read_csv("London_11.csv")
# Load the entire pipeline
pipe = pickle.load(open("random_forest_pipeline.pkl", 'rb'))


@app.route('/')
def index():
    countys = sorted(data['County'].unique())
    locations = sorted(data['Location'].unique())
    house_types = sorted(data['House_Type'].unique())
    area_classifiers = sorted(data['Area_Classifier'].unique())
    central_london_flags = sorted(data['Central_London_Flag'].unique())

    return render_template('house_price_form.html', locations=locations, house_types=house_types, acls=area_classifiers, lndn=central_london_flags, countys=countys)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data

    House_Type = request.form.get('House_Type')
    County = request.form.get('County')
    Location = request.form.get('location')
    Central_London_Flag = request.form.get('Central_London_Flag')
    Area_Classifier = request.form.get('Area_Classifier')
    Bedrooms = int(request.form.get('Bedrooms'))
    Bathrooms = int(request.form.get('Bathrooms'))
    Area = float(request.form.get('Area'))
    Receptions = int(request.form.get('Receptions'))
    
    # Construct a DataFrame with the same column names as the training data
    input_data = {
        'House_Type': [House_Type], 
        'County': [County], 
        'Central_London_Flag': [Central_London_Flag],
        'Area_Classifier': [Area_Classifier], 
        'Bedrooms': [Bedrooms], 
        'Bathrooms': [Bathrooms], 
        'Area': [Area], 
        'Receptions': [Receptions],
        'Location' : [Location]
    }
    input_df = pd.DataFrame(input_data)
    
    # Use the pipeline for prediction
    prediction = pipe.predict(input_df)[0]
    return str(np.round(prediction, 2))

@app.route('/get-locations/<county>', methods=['GET'])
def get_locations(county):
    filtered_locations = data[data['County'] == county]['Location'].unique().tolist()
    return jsonify(filtered_locations)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)

