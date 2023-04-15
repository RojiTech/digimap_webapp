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
def compute_dark_channel(img,radius=7):
    h,w=img.shape[:2]
    window_size=2*radius+1
    b, g, r = cv2.split(img)
    bgr_min_img = cv2.min(cv2.min(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size,window_size))
    dark_channel_img = cv2.erode(bgr_min_img, kernel)
    return dark_channel_img

def compute_atmosphere_light(img,dark_channel_img):
    h,w=dark_channel_img.shape[:2]
    num_of_candiate=int(0.001*h*w)
    dark_channel=dark_channel_img.reshape(-1,1)[:,0 ]
    arg_sorted=np.argsort(dark_channel)[::-1]
    img=img.astype(np.float32)
    atmosphere_light=np.zeros((3,))
    for i in range(num_of_candiate):
        index=arg_sorted[i]
        row_index=index//w
        col_index=index%w
        for c in range(3):
            atmosphere_light[c]=max(atmosphere_light[c],img[row_index,col_index][c])
    return atmosphere_light


def compute_transmission_rate(img,atmosphere_light_max,omega,dark_channel_img,guided_filter_radius,epsilon):
    h,w=img.shape[:2]
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    zero_mat=np.zeros((h,w))
    transmition_rate_est=cv2.max(zero_mat,np.ones_like(zero_mat)-omega*dark_channel_img/atmosphere_light_max)
    transmission_rate = guied_filter(img_gray,transmition_rate_est, guided_filter_radius, epsilon)
    #transmission_rate = cv2.GaussianBlur(transmition_rate_est, (guided_filter_radius, guided_filter_radius), epsilon, borderType=cv2.BORDER_REPLICATE)
    return transmission_rate


def guied_filter(I,P,radius,epsilon):
    window_size=2*radius+1
    meanI=cv2.blur(I,(window_size,window_size))
    meanP=cv2.blur(P,(window_size,window_size))
    II=I**2
    IP=I*P
    corrI=cv2.blur(II,(window_size,window_size))
    corrIP=cv2.blur(IP,(window_size,window_size))
    varI=corrI-meanI**2
    covIP=corrIP-meanI*meanP
    a=covIP/(varI+epsilon)
    b=meanP-a*meanI
    meanA=cv2.blur(a,(window_size,window_size))
    meanB=cv2.blur(b,(window_size,window_size))
    transmission_rate=meanA*I+meanB
    return transmission_rate

def dehaze(img,radius=7,omega=0.95,epsilon=0.0001,guided_filter_radius=25,transmission_thresh=0.1):
    dark_channel_img=compute_dark_channel(img,radius)
    
    atmosphere_light=compute_atmosphere_light(img,dark_channel_img)
    transmission_rate=compute_transmission_rate(img,np.max(atmosphere_light),omega,dark_channel_img,guided_filter_radius=guided_filter_radius,epsilon=epsilon)
    transmission_rate[transmission_rate<transmission_thresh]=transmission_thresh
    
    dehaze_img=np.zeros_like(img)
    for c in range(3):
        dehaze_img[:,:,c]=(img[:,:,c]-atmosphere_light[c])/transmission_rate+atmosphere_light[c]
    dehaze_img[dehaze_img>1]=1
    dehaze_img[dehaze_img<0]=0
    return dehaze_img

def dehaze_function(img):
    
    img=img.astype(np.float32)
    img/=255
    dehaze_img=dehaze(img,radius=3)


    return dehaze_img*255

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
        #dehaze_path = os.path.join(
        #    basepath, 'dehazed_images', secure_filename(dehaze_fname))
        dehaze_path = os.path.join(
            basepath, app.config['DEHAZED_FOLDER'], dehaze_fname)
        fname=os.path.basename(dehaze_path)
        print(fname)
        cv2.imwrite(dehaze_path,dehaze_final)
        return render_template('predict.html',file_name=file_name, dehazed_file=fname)

    return ""

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['DEHAZED_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)
