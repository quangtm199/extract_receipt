
from fastapi import FastAPI, File,UploadFile,Request,Form
from typing import Optional
import io
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from pydantic import BaseModel
from flask import Markup
import argparse
import numpy as np
import uvicorn
from layoutlmv2 import predict, get_model
from starlette.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles #new
from backend.models import load_text_detect, load_text_recognize, load_saliency
import requests
import io
import json
import urllib
import cv2
from typing import Optional
templates = Jinja2Templates(directory='templates/')
app = FastAPI()
def configure_static(app):  #new
    app.mount("/static", StaticFiles(directory="static"), name="static")
configure_static(app)

class Item(BaseModel):
    source: str
model,processor=get_model()
net = load_saliency()
detector = load_text_recognize()
text_detector = load_text_detect()


# cfg,modelSpade,tokenizer=get_modelSpade()
UPLOAD_FOLDER= "static"
import os
from PIL import Image
@app.get('/')
def form_post(request: Request):
    
    return templates.TemplateResponse('indexfast.html', context={'request': request})

@app.get('/mlbigdata/detect_person/')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('indexfast.html', context={'request': request,'result': ""})
@app.get('/mlbigdata/detect_person/file')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('indexfast.html', context={'request': request,'result': ""})
def download_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

@app.post("/mlbigdata/detect_person/file")
async def home_page(request: Request,width: str = Form("400"),height: str = Form("200"),file: UploadFile = File(...),url_image: str = Form("100") ):
    # try:
    # Lấy file gửi lên
    if url_image!="100":
        print("url")
        image_url=url_image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        input_image1=image
#         image = requests.get(image_url).content
#         image='static/pic1.jpg'
#         with open('static/pic1.jpg', 'wb') as handle:
#             handle.write(image)
    else:    
        print("image")
        image = await file.read()
        input_image = Image.open(io.BytesIO(image))
        input_image1=input_image.convert('RGB')
        input_image1 = cv2.cvtColor(np.array(input_image1), cv2.COLOR_BGR2RGB)
    
    if image:
        print(width,height)
         
        input_image1 = np.array(input_image1)
        input_image2,chuoi_result=predict(input_image1,model,processor,net,detector,text_detector)
        print(input_image2)
        # input_image2.save("3.jpg")
        image_path="static/"+file.filename[:-4]+"old.jpg"
        path1=file.filename[:-4]+".jpg"
        cv2.imwrite(image_path,input_image1)
        cv2.imwrite("static/"+path1,input_image1)
        path=file.filename[:-4]+"predict.jpg"
        input_image2.save("static/"+path)
        return templates.TemplateResponse('indexfast.html', context={'request': request, 'result': path,'path123' : path1,'path1234' : path , 'image_path':image_path,'chuoitextocr':Markup(chuoi_result) })
@app.post("/file")
async def home_page1(request: Request,width: str = Form("400"),height: str = Form("200"),file: UploadFile = File(...),url_image: str = Form("100") ):
    # try:
    # Lấy file gửi lên
    if url_image!="100":
        print("url")
        image_url=url_image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        input_image1=image
#         image = requests.get(image_url).content
#         image='static/pic1.jpg'
#         with open('static/pic1.jpg', 'wb') as handle:
#             handle.write(image)
    else:    
        print("image")
        image = await file.read()
        input_image = Image.open(io.BytesIO(image))
        input_image1=input_image.convert('RGB')
        input_image1 = cv2.cvtColor(np.array(input_image1), cv2.COLOR_BGR2RGB)
    
    if image:
        print(width,height)
         
        input_image1 = np.array(input_image1)
        input_image2,chuoi_result=predict(input_image1,model,processor,net,detector,text_detector)
        print(input_image2)
        # input_image2.save("3.jpg")
        image_path="static/"+file.filename[:-4]+"old.jpg"
        path1=file.filename[:-4]+".jpg"
        cv2.imwrite(image_path,input_image1)
        cv2.imwrite("static/"+path1,input_image1)
        path=file.filename[:-4]+"predict.jpg"
        input_image2.save("static/"+path)
        return templates.TemplateResponse('indexfast.html', context={'request': request, 'result': path,'path123' : path1,'path1234' : path , 'image_path':image_path,'chuoitextocr':Markup(chuoi_result) })

# @app.post("/mlbigdata/detect_person/file")
# async def home_page(request: Request,url_image: str = Form("100"),file: UploadFile = File(...) ):
#     return {"url_image": url_image}
#     # try:
#     # Lấy file gửi lên
#     image = await file.read()
#     if image:
#         print(width,height)
#         input_image = Image.open(io.BytesIO(image))

#         input_image1=input_image.convert('RGB')
#         input_image1 = cv2.cvtColor(np.array(input_image1), cv2.COLOR_BGR2RGB) 
#         input_image1 = np.array(input_image1)
#         input_image2,chuoi_result=predict(input_image1,model,processor,net,detector,text_detector)
#         print(input_image2)
#         # input_image2.save("3.jpg")
#         image_path="static/"+file.filename[:-4]+"old.jpg"
#         path1=file.filename[:-4]+".jpg"
#         cv2.imwrite(image_path,input_image1)
#         cv2.imwrite("static/"+path1,input_image1)
#         path=file.filename[:-4]+"predict.jpg"
#         input_image2.save("static/"+path)

#         return templates.TemplateResponse('indexfast.html', context={'request': request, 'result': path,'path123' : path1,'path1234' : path , 'image_path':image_path ,'chuoitextocr':Markup(chuoi_result) })

    
if __name__=="__main__":
   
    uvicorn.run("main:app", port = 4002, host = "172.26.33.214",reload=True)
