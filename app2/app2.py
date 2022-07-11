from joblib import load
import numpy as np
from flask import Flask, render_template,request
app = Flask(__name__)
model=load("app2/static/profit.pk")

@app.route('/',methods=["GET","POST"])
def index():
  if request.method=="POST":
    ran=int(request.form.get("ran"))
    adm=int(request.form.get("adm"))
    ms=int(request.form.get("ms"))
    ste=request.form.get("ste")

    en=model['state_enc']
    sc=model['scaler']
    mo=model['model']

    dummy=en.transform([[ste]]).toarray()
    inp=np.array([ran,adm,ms]).reshape(1,-1)
    inp=np.concatenate((dummy,inp),axis=1)
    scale_input=sc.transform(inp)
    profit=mo.predict(scale_input)[0]
    return render_template('index.html',profit=profit)
  return render_template('index.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 