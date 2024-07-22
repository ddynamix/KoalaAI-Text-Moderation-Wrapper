from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import hf_model

app = FastAPI()


class TextRequest(BaseModel):
    text: str


@app.post("/predict/")
def get_prediction(request: TextRequest):
    try:
        predictions = hf_model.predict(request.text)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
