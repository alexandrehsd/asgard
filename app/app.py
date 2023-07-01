import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

t1, t2 = st.columns(2)
with t1:
    st.markdown('## Sustainable Development Goal Classifier for Academic Papers')

with t2:
    st.write("")
    st.write("")
    # st.image("./app/dca.png", width=300, output_format="PNG")
    st.write("")
    st.write("""
             **Department of Computer Engineering and Automation** \\
             Federal University of Rio Grande do Norte - Brazil
             """)

st.write("")
st.markdown("""
The Sustainable Development Goal (SDG) classifier, non-creatively nicknamed ASGARD, is designed to categorize academic research papers into one or more of the inspiring SDGs.

With this remarkable technology, we can unlock the immense potential of academic research by connecting it directly to the 
global agenda for sustainable development. 
By leveraging the power of deep learning, our classifier can swiftly analyze vast amounts of data, enabling researchers, 
policymakers, and organizations to navigate through the extensive landscape of research papers and pinpoint 
those that align with the specific SDGs they are passionate about.

Imagine the possibilities this opens up! Our model empowers individuals and institutions to contribute effectively to the SDGs, 
promoting sustainable development in various areas such as education, health, poverty alleviation, climate action, and so much more. 
By accurately classifying research papers, we can accelerate the discovery of innovative solutions, 
foster collaboration, and drive positive change towards a brighter and more sustainable future.

for more details related to the SDGs, please visit https://sdgs.un.org/goals.
""")


with st.sidebar.expander("Click to learn more about ASGARD"):
    st.markdown(f"""
    The United Nations created the 17 Sustainable Development Goals (SDGs) to promote
    environmental protection, economic growth, and social justice. In this scenario, science is crucial
    to solving the challenges addressed by the SDGs.
    
    ASGARD is a multi-label Deep-Learning Classifier that maps scientific publications to the SDGs. 
    The only information ASGARD needs from the researcher is the title of their paper. Currently, ASGARD
    does not make predictions for the SDG 17 (Partnerships for the Goals) because of unavailability in the data
    of this label in the training data.

    Data source: The data used for model training was obtained manually from the **Scopus database** through the SciVal tool. 
    In the data collection process, only articles published between 2019 and 2022 and related to at least one SDG were considered.
    In terms of volume, about **800k academic paper titles** were used to train the classifier.
    
    The model card and source code of this project is available on [GitHub](https://github.com/alexandrehsd/asgard).
    
    *Released on July 1st, 2023*  
    """)

st.markdown('## Try it out')

# put a textarea and button on the page
text = st.text_area("Input the title of your paper below.")

# when the button is clicked, write the message
if st.button("Submit"):

    payload = {}
    payload["text"] = text

    # send text to the api
    # response = requests.post("http://api:8000/api", json=payload)

    # use this outside docker container
    response = requests.post("http://localhost:8000/api", json=payload)
    
    # write the response to the page
    # st.write(response.json()['message'])
    # example: Law enforcement effects on marine life preservation in the South Pacific
    
    threshold = 0.5
    probabilities = np.fromstring(response.json()['message'][2:-2], sep=" ")

    probabilities = pd.DataFrame(data={"Probability": probabilities,
                                       "Sustainable Development Goals": [f"SDG {i+1}" for i in range(16)],
                                       "Classes": ["Positive" if prob else "Negative" for prob in (probabilities >= 0.5)],
                                       "Order": [i for i in range(16)]},
                                 )
    
    bars = alt.Chart(probabilities).mark_bar().encode(
        x=alt.X('Sustainable Development Goals:O', sort=None), 
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=(0, 1.0))),
        color=alt.condition(
            alt.datum.Probability >= threshold,  # If the probability is higher or equal 0.5 returns True,
            alt.value('orange'),     # which sets the bar orange.
            alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
            ),
        order=alt.Order('Order:Q')
        )
    
    st.altair_chart((bars).properties(width=600), use_container_width=True)
    