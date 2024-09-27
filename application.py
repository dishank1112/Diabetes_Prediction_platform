from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

dt_model = pickle.load(open('./models/dt.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        bmi = float(request.form.get('bmi'))
        bp = float(request.form.get('bp'))
        s1 = float(request.form.get('s1'))
        s2 = float(request.form.get('s2'))
        s3 = float(request.form.get('s3'))
        s4 = float(request.form.get('s4'))
        s5 = float(request.form.get('s5'))
        s6 = float(request.form.get('s6'))
    
        result = dt_model.predict([[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]])

        return render_template('home.html', results=result[0])
    else:
        render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
