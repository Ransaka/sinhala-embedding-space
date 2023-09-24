import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from embeddings.embeddings import load_model
from sentence_transformers import  util
# Create sample data
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 22, 35]
})

# Sample PNG file
image = Image.open('plots\clusters.png')

# Sample HTML chart
chart_data = pd.read_csv(r"data\top_cluster_dataset.csv",dtype={'Headline': str, 'x': np.float64, 'y': np.float64, 'labels': str})

# Create a Streamlit app
st.set_page_config(page_title="Sample Webpage", page_icon=":bar_chart:")

# Define tabs
tabs = ["Search", "Clustering Results"]
selected_tab = st.sidebar.radio("Select a Tab", tabs)

# Main content
if selected_tab == "Search":
    sample_sentences = chart_data['Headline'].sample(10, random_state=1).tolist()
    st.title("Calculate Sentences Similarity")
    # select model to use dropdown
    st.subheader("Select a model to use")
    model_list = ["Ransaka/SinhalaRoberta","keshan/SinhalaBERTo"]
    selected_model = st.selectbox("Select Model", model_list)
    model = load_model(selected_model)
    
    sentence1 = st.text_input("Enter Sentence 1", "")
    sentence2 = st.text_input("Enter Sentence 2", "")

    if sentence1 and sentence2:
        # add button to calculate similarity
        if st.button("Calculate Similarity"):
            with st.spinner('Calculating Similarity...'):
                # Calculate similarity
                similarity = util.pytorch_cos_sim(model.encode(sentence1), model.encode(sentence2))[0][0]
                if similarity > 0.7:
                    st.success(f"Sentences are similar (Score: {similarity:.3f})")
                elif similarity > 0.5:
                    st.warning(f"Sentences are somewhat similar (Score: {similarity:.3f})")
                else:
                    st.error(f"Sentences are not similar (Score: {similarity:.3f})")
    else:
        st.write("Enter two sentences to calculate similarity. Or start with sample sentences below.")
        # change radio button to randomize sentences and show sample sentences
        if st.button("Randomize Sentences"):
            sample_sentences = chart_data['Headline'].sample(10).tolist()
        for sentence in sample_sentences:
            # show sample sentences in small font
            st.write(sentence)

elif selected_tab == "Clustering Results":
    st.title("Clustering Results Tab")
    
    # Display PNG image
    st.subheader("Static PNG File")
    st.image(image, use_column_width=False, caption='Static PNG File',width=750)
    
    altair_chart = alt.Chart(chart_data).mark_circle().encode(
        x='x',
        y='y',
        color='labels',
        tooltip='Headline'
    ).properties(
        width=750,
        height=500
    ).interactive()
    # Display chart
    st.subheader("Interactive Chart for top clusters")
    st.altair_chart(altair_chart, use_container_width=False, theme="streamlit")
    
    # Dropdown functionality to update DataFrame
    st.subheader("Select a cluster")
    unique_clusters = chart_data['labels'].unique().tolist()
    selected_value = st.selectbox("Select Value", unique_clusters)
    
    # Filter and display results based on selected cluster
    if selected_value:
        filtered_data = chart_data[chart_data['labels'].str.contains(selected_value, case=False)].sample(10)[['Headline']].reset_index(drop=True)
        st.dataframe(filtered_data,width=750)
    else:
        st.write("Select a cluster to display results.")

