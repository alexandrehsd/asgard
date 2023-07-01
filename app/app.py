import requests
import numpy as np
import streamlit as st

st.title("Hello World")
st.write("This is a Streamlit app")

# put a textarea and button on the page
text = st.text_area("Input your text here")

# when the button is clicked, write the message
if st.button("Submit"):

    payload = {}
    payload["text"] = text

    # send text to the api
    response = requests.post("http://api:8000/api", json=payload)

    # use this outside docker container
    # response = requests.post("http://localhost:8000/api", json=payload)
    
    # write the response to the page
    st.write(response.json()['message'])
    
    probabilities = np.fromstring(response.json()['message'][2:-2], sep=" ")

    # plot a bar chart
    st.bar_chart(probabilities)