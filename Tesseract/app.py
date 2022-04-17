from re import search
from PIL.Image import Image
from flask import Flask, render_template, request,url_for
from pytesseract.pytesseract import Output
from werkzeug.utils import redirect
import PyTesseract
import cv2
import os
from skimage import io

op = ""
pic = ""
app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def predict():
 
    if request.method == 'POST':
        global pic
        image = request.files['data']
        pic = io.imread(image)
        os.chdir(r"D:\Projects\OCR-NLP\Tesseract\static")
        cv2.imwrite("local.jpg",pic)
        global op
        op = PyTesseract.main(image,"bert","")
        return redirect('/')
       
    else:
        last = op
        op = ""
        return render_template('GUI.html',Output = last,image = "/static/local.jpg")
  

if __name__ == '__main__':
    app.debug = True
    app.run(port=8000)
  
