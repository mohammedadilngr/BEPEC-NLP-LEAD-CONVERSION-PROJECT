import numpy as np
import pickle
import flask
from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__, template_folder='templates')

tv_t1= pickle.load(open("tv_t1.pkl", "rb"))
model_t1 = pickle.load(open("model_t1.pkl", "rb"))
tf1= pickle.load(open("tf1.pkl", "rb"))
tf2 = pickle.load(open("tf2.pkl", "rb"))
tf3= pickle.load(open("tf3.pkl", "rb"))
model_t2 = pickle.load(open("model_t2.pkl", "rb"))



@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


def value_predictor(user_input):
    vector = tv_t1.transform(user_input).toarray()
    output = model_t1.predict(vector)
    return output

def value_predictor1(user_input1,user_input2,user_input3):
    vector1 = tf1.transform(user_input1).toarray()
    vector2 = tf2.transform(user_input2).toarray()
    vector3 = tf3.transform(user_input3).toarray()
    vector=np.concatenate((vector1,vector2,vector3),axis=1)
    output = model_t2.predict(vector)
    return output


@app.route('/result', methods = ['POST'])
def result():

    if request.method == 'POST':
        message = request.form['Message']

        if message=="":
             prediction = "please copy paste the msg/sms"


        else:
            data = [message]
            result = value_predictor(data)
            
    return render_template("after1.html", prediction=result)

@app.route('/result1', methods = ['POST'])
def result1():

    if request.method == 'POST':
        message1 = request.form['Message1']
        message2 = request.form['Message2']
        message3 = request.form['Message3']

        if message1=="":
             prediction = "please copy paste the msg/sms"
        elif message2=="":
             prediction = "please copy paste the msg/sms"
        elif message3=="":
            prediction = "please copy paste the msg/sms"


        else:
            data1 = [message1]
            data2=[message2]
            data3=[message3]
            result = value_predictor1(data1,data2,data3)
            
            
   

    return render_template("after2.html", prediction=result)



if __name__=="__main__":
    app.run()  


