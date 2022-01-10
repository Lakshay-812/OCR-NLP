from flask import Flask, render_template, request
from werkzeug.utils import redirect
import cv2
import os
from skimage import io
import htr
op = ""
pic = ""
app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def predict():
 
    if request.method == 'POST':
        global pic
        image = request.files['data']
        pic = io.imread(image)
        os.chdir(r"C:\Users\AMIT JAIN\Desktop\CSD 350\NLP-OCR\OCR-NLP\Handwriting OCR\src\static")
        cv2.imwrite("local.jpg",pic)
        global op
        op = htr.main("infer","bestpath")
        #op = htr.main("infer","wordbeamsearch")
        return redirect('/')
       
    else:
        last = op
        op = ""
        return render_template('GUI.html',Output = last,image = "/static/local.jpg")
  

if __name__ == '__main__':
    app.debug = True
    app.run()
  
