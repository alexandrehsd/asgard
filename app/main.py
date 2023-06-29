import requests
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
    response = requests.post("http://localhost:8000/api", json=payload)

    # write the response to the page
    st.write(response.json()['message'])