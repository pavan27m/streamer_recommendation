# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

app = Flask(__name__)
# Load the regression model
model = joblib.load(open('model.pickle','rb'))
scaling = joblib.load(open('scaler.pickle','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    duration = int(request.form['duration'])
    count = int(request.form['count'])
    ios = int(request.form['ios'])
    android = int(request.form['android'])
    durationGTE5sec = int(request.form['fivesec'])
    durationGTE2min = int(request.form['twomin'])
    durationGTE10min = int(request.form['tenmin'])
    receivePointEstimated = int(request.form['point'])
    maxLiveViewerCount = int(request.form['maxviewer'])

    arr = [[duration,count,ios,android,durationGTE5sec,durationGTE2min,durationGTE10min,receivePointEstimated,maxLiveViewerCount]]
    z = scaling.transform(arr)


    import functools
    import operator

    
    z = functools.reduce(operator.concat, z)
    duration = [z[0]]  
    count = [z[1]]
    ios = [z[2]]
    android = [z[3]] 
    durationGTE5sec = [z[4]]
    durationGTE2min = [z[5]]
    durationGTE10min = [z[6]] 
    receivePointEstimated = [z[7]]
    maxLiveViewerCount = [z[8]]


    data = [duration + count + ios + android + durationGTE5sec + durationGTE2min + durationGTE10min + receivePointEstimated + maxLiveViewerCount]


    if (duration == 0 or count ==0 or ios == 0 or android == 0 or durationGTE5sec ==0 or durationGTE2min == 0 or durationGTE10min == 0 or receivePointEstimated ==0 or maxLiveViewerCount == 0):
        my_prediction = [0]
    else:
        my_prediction = model.predict(data)


    if (my_prediction == 1):
        prediction = 'Good'
    else:
        prediction = 'Bad'

    return render_template('index.html', prediction_text='The Stremer Performance {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)