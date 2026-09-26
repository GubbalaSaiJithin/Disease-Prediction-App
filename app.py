from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
from symptom_model import load_artifact, predict_symptoms

st.set_page_config(page_title='Symptom Classification Demo', layout='centered')
st.title('Symptom Classification Demo')
st.caption('Educational machine-learning project. Not a diagnostic tool or medical advice.')

@st.cache_resource
def get_artifact():
    return load_artifact()

try:
    artifact = get_artifact()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

options = list(artifact['model'].named_steps['features'].get_feature_names_out())
selected = st.multiselect('Select symptoms', options=options)
if st.button('Classify symptoms'):
    try:
        results = predict_symptoms(selected, artifact)
        st.subheader('Model classification')
        st.write(results[0]['label'])
        st.dataframe(pd.DataFrame(results), hide_index=True)
        st.caption('Model scores are relative classifier outputs, not calibrated probabilities of illness.')
    except ValueError as error:
        st.error(str(error))

with st.expander('Evaluation and limitations'):
    metrics = artifact['metrics']
    st.write(f"Training: {metrics['train_rows']} unique symptom sets; test: {metrics['test_rows']}.")
    st.write(f"Held-out accuracy: {metrics['test_accuracy']:.1%}; macro F1: {metrics['test_macro_f1']:.3f}.")
    st.write(metrics['limitation'])
