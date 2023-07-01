import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import uvicorn
import tensorflow as tf
from fastapi import FastAPI
from utils import Payload, ModelsLoaderSingleton

from absl import logging
logging.set_verbosity(logging.ERROR)

tf.get_logger().setLevel('ERROR')

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Load the models
    """
    ModelsLoaderSingleton.instance()


@app.post("/api")
async def api(payload: Payload):

    # get the text from the payload
    text = payload.text

    predicted = ModelsLoaderSingleton.instance().model.predict([text])

    # print(predicted[0])
    # example of input text: Law enforcement effects on marine life preservation
    # do something with the text
    return {"message": str(predicted)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)