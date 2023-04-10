import sys
import os, shutil
import glob
import re
import numpy as np
import cv2
import math

from flask import Flask,flash, request, render_template,send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['DEHAZED_FOLDER'] = 'dehazed_images'
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dehazed_images/<filename>')
def dehazed_img(filename):
    
    return send_from_directory(app.config['DEHAZED_FOLDER'], filename)
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def dehaze_function(img):
    
    img_8bit = cv2.convertScaleAbs(img)
    I = img_8bit.astype('float32')/255;

    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(I,te);
    J = Recover(I,t,A,0.1);
    J = (J * 255).astype('uint8')

    return J

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']
        style = request.form.get('style')
        print(style)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        
        f.save(file_path)
        file_name=os.path.basename(file_path)
        
        # reading the uploaded image
        
        img = cv2.imread(file_path)
        dehaze_fname =file_name + "_dehaze_image.jpg"
        dehaze_final = dehaze_function(img)
        dehaze_path = os.path.join(
            basepath, 'dehazed_images', secure_filename(dehaze_fname))
        fname=os.path.basename(dehaze_path)
        print(fname)
        cv2.imwrite(dehaze_path,dehaze_final)
        return render_template('predict.html',file_name=file_name, dehazed_file=fname)

    return ""

if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)
