# from PRE import predict
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, Form
import json
import uvicorn
from predict import Predictor
import tempfile
import cv2
import os
import connector
app = FastAPI()
predictor = Predictor()

@app.get("/")
async def root():
    return {"message": "Hello World"}


app = FastAPI()

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}

@app.post("/sample/")
async def get_sample(myJson: str = Form(...)):
    specie = json.loads(myJson)["zhong"]
    samples = connector.get_sample(specie)
    return {"samples": samples}

@app.post("/uploadfile/") 

async def create_upload_file(myJson: str = Form(...),file: UploadFile = File(...)):
    contents = await file.read()
    print(myJson)
    with open(file.filename,'wb') as f:
        f.write(contents)
    with open(file.filename.split(".")[0]+".json","w",encoding='utf-8') as json_f:
        json.dump(myJson, json_f)
    #jstring = json.dumps(myJson)
    #print(jstring)
    connector.upload_temp(file.filename,myJson)
    
    return {"filename": file.filename,"file_type": file.content_type}
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    with open("./img/"+file.filename,"wb") as f:
        f.write(contents)
        result = predictor.predict("./img",file.filename,"C:/Users/Administrator/Desktop/result")
    
    return result  
if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000, reload=False, debug=False)
