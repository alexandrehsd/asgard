import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


st.markdown("""
<style>
.small-font {
    font-size:12px !important;
}
</style>
""", unsafe_allow_html=True)

SDG = {
    1: {"name": "SDG 1",
        "title": "No Poverty",
        "description": "End poverty in all its forms everywhere",
        "details": "https://www.undp.org/sustainable-development-goals/no-poverty"},
    2: {"name": "SDG 2",
        "title": "Zero Hunger",
        "description": "End hunger, achieve food security and improved nutrition, and promote sustainable agriculture",
        "details": "https://www.undp.org/sustainable-development-goals/zero-hunger"},
    3: {"name": "SDG 3",
        "title": "Good Health and Well-Being",
        "description": "Ensure healthy lives and promote well-being for all at all ages",
        "details": "https://www.undp.org/sustainable-development-goals/good-health"},
    4: {"name": "SDG 4",
        "title": "Quality Education",
        "description": "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all",
        "details": "https://www.undp.org/sustainable-development-goals/quality-education"},
    5: {"name": "SDG 5",
        "title": "Gender Equality",
        "description": "Achieve gender equality and empower all women and girls",
        "details": "https://www.undp.org/sustainable-development-goals/gender-equality"},
    6: {"name": "SDG 6",
        "title": "Clean Water and Sanitation",
        "description": "Ensure availability and sustainable management of water and sanitation for all",
        "details": "https://www.undp.org/sustainable-development-goals/clean-water-and-sanitation"},
    7: {"name": "SDG 7",
        "title": "Affordable and Clean Energy",
        "description": "Ensure access to affordable, reliable, sustainable and modern energy for all",
        "details": "https://www.undp.org/sustainable-development-goals/affordable-and-clean-energy"},
    8: {"name": "SDG 8",
        "title": "Decent Work and Economic Growth",
        "description": "Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all",
        "details": "https://www.undp.org/sustainable-development-goals/decent-work-and-economic-growth"},
    9: {"name": "SDG 9",
        "title": "Industry, Innovation and Infrastructure",
        "description": "Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation",
        "details": "https://www.undp.org/sustainable-development-goals/industry-innovation-and-infrastructure"},
    10: {"name": "SDG 10",
         "title": "Reduced Inequalities",
         "description": "Reduce income inequality within and among countries",
         "details": "https://www.undp.org/sustainable-development-goals/reduced-inequalities"},
    11: {"name": "SDG 11",
         "title": "Sustainable Cities and Communities",
         "description": "Make cities and human settlements inclusive, safe, resilient, and sustainable",
         "details": "https://www.undp.org/sustainable-development-goals/sustainable-cities-and-communities"},
    12: {"name": "SDG 12",
         "title": "Responsible Consumption and Production",
         "description": "Ensure sustainable consumption and production patterns",
         "details": "https://www.undp.org/sustainable-development-goals/responsible-consumption-and-production"},
    13: {"name": "SDG 13",
         "title": "Climate Action",
         "description": "Take urgent action to combat climate change and its impacts by regulating emissions and promoting developments in renewable energy",
         "details": "https://www.undp.org/sustainable-development-goals/climate-action"},
    14: {"name": "SDG 14",
         "title": "Life Below Water",
         "description": "Conserve and sustainably use the oceans, seas and marine resources for sustainable development",
         "details": "https://www.undp.org/sustainable-development-goals/below-water"},
    15: {"name": "SDG 15",
         "title": "Life on Land",
         "description": "Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss",
         "details": "https://www.undp.org/sustainable-development-goals/life-on-land"},
    16: {"name": "SDG 16",
         "title": "Peace, Justice and Strong Institutions",
         "description": "Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels",
         "details": "https://www.undp.org/sustainable-development-goals/peace-justice-and-strong-institutions"}
}


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
The Sustainable Development Goal (SDG) classifier (a.k.a ASGARD), is designed to categorize academic research papers into one or more of the inspiring SDGs.

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
    st.markdown("""
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
    
    *Released on August 31st, 2023*  
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
    probabilities = np.round(np.fromstring(response.json()['message'][2:-2], sep=" "), 3)
    # st.write(str(probabilities))
    
    data = pd.DataFrame(data={"Probability": probabilities,
                              "Sustainable Development Goals": [f"SDG {i+1}" for i in range(16)],
                              "SDG": [f"SDG {i+1}" for i in range(16)],
                              "Classes": ["Positive" if prob else "Negative" for prob in (probabilities >= 0.5)],
                              "Order": [i+1 for i in range(16)],
                              "Description": [SDG[i+1]["title"] for i in range(16)]
                              },
                        )
    
    labels = data.loc[data["Classes"] == "Positive", "Order"].values.tolist()
    
    # CLASSIFICATION REPORT
    st.markdown('## Classification Report')
    
    if len(labels) == 0:
        st.write("ASGARD could not relate your input to any of the Sustainable Development Goals. :confused:")
        st.write("_Tip: You can try again using other terms. Try to be more specific and key-word oriented._")
    else:
        st.write("ASGARD linked your input text to the following Sustainable Development Goal(s):")
        for label in labels:
            st.write(f"##### {SDG[label]['name']}: {SDG[label]['title']}")
            st.write(f"**Description**: {SDG[label]['description']}. [Learn more]({SDG[label]['details']}).")
    
    
    # MORE DETAILS ABOUT THE PREDICTION
    with st.expander("Expand for More Details About the Classification"):
        
        # Create the custom color scale
        color_scale = alt.Scale(
            domain=[0, 0.5, 0.8, 1],
            range=['steelblue', 'yellow', 'orange']
        )

        # Create the chart
        bars = alt.Chart(data).mark_bar().encode(
            x=alt.X('Sustainable Development Goals:N', sort=None, axis=alt.Axis(labelFontSize=9, labelAngle=0)), 
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=(0, 1.0))),
            color=alt.Color(
                'Probability:Q',
                scale=color_scale,
                legend=alt.Legend(title='Probability', labelFontSize=10, titleFontSize=12, symbolSize=150)
            ),
            order=alt.Order('Order:Q'),
            tooltip=[alt.Tooltip("SDG:N"), 
                     alt.Tooltip("Description:N"), 
                     alt.Tooltip("Probability:Q")]
            )
        
        # Add text mark for y-axis values
        text = bars.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,  # Adjust the vertical offset as needed
            fontSize=10
        ).encode(
            text=alt.Text('Probability:Q', format='.3f')
        )
        
        # Combine the bars and text marks and Set the title
        chart = (bars + text).properties(
            width=600, 
            title='Predicted Probability of Text-SDG Association'
            )

        st.altair_chart(chart, use_container_width=True)
        
        st.write("""
            <p class="small-font"><em>ASGARD provides the probability for each label's association with the Sustainable Development Goals (SDGs). 
            A probability threshold of 0.5 is used to determine whether a given input is related to an SDG or not. 
            Consequently, the higher the output probability towards 1.0, the greater the model's certainty regarding  
            that prediction. It is important to bear this in mind, as output probabilities ranging from 0.5 to 0.8 
            are considered as positive classifications with low confidence.</em></p>
            """,
            unsafe_allow_html=True)
        
    # EXPLAINABLE AI
    