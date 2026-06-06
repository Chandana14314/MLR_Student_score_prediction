import flask
from flask import Flask,render_template,request
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression

with open("MLR_Exam.pkl","rb") as f:
    m = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def initial_fun():
    return render_template("index.html")

@app.route("/predict",methods = ['GET','POST'])
def fun3():
    a = [float(i) for i in request.form.values()]
    b = [np.array(a)]
    sol = m.predict(b)[0]
    return render_template("index.html" , Prediction = sol)


if __name__ == "__main__":
    app.run(debug=True)