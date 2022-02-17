# main.py
# use this API to develop a data collector project!!
from fastapi import FastAPI
app = FastAPI()
# package name must be : src!!
# running command : uvicorn --port 5000 --host 127.0.0.1 src.main:app --reload

@app.get("/")
def hello():
    return {"message":"Hello TutLinks.com"}

@app.get("/myRoute")
def myroute():
    return {"success":"what is this"}
