from turtle import distance
from joblib import load
import numpy as np
from flask import Flask, render_template,request
app = Flask(__name__)
model=load("supervised/regression/car_dist.joblib")

@app.route('/',methods=["GET","POST"])
def index():
  if request.method=="POST":
    speed=int(request.form.get("spd"))
    inp=np.array([speed])
    inp=inp.reshape(-1,1)
    distance=model.predict(inp)
    return render_template('index.html',distance=distance)
  return render_template('index.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 