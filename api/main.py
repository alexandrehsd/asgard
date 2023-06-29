import uvicorn
import tensorflow as tf
from fastapi import FastAPI
from api.utils import Payload, ModelsLoaderSingleton

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

    print(predicted)
    
    # do something with the text
    return {"message": str(predicted)}


if __name__ == "__main__":
    # 18.117.99.111	- Public IPv4 address
    uvicorn.run(app, host="localhost", port=8000)