import pickle
import numpy as np
from flask import render_template, jsonify, Flask, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexpage.html', form='visible', sat='hidden')


@app.route('/prediction', methods=['POST'])
def predict():
    traveltype = request.form['traveltype']
    wifi = int(request.form['wifi'])
    onlineBoarding = int(request.form['onlineBoarding'])
    seatComfort = int(request.form['seatComfort'])
    checkin = int(request.form['checkin'])

    if traveltype == "Personal Travel":
        travel = 1
    else:
        travel = 0

    features = [travel, wifi, onlineBoarding, seatComfort, checkin]
    
    final_features = [np.array(features)]
    # Normalize Features
    X_test = scaler.transform(final_features)
    
    prediction = model.predict(X_test)

    if prediction:
        label = 'Satisfied'
    else:
        label = 'Neutral/ Dissatisfied'

    return render_template('indexpage.html', text=label, form='hidden', sat='visible')
    

@app.route('/clustering')
def clustering():
    return render_template('clusters.html')



if __name__ == "__main__":
    app.run(debug=True)