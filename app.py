import streamlit as st
import pickle
import numpy as np

# Set the page title and include a custom logo
st.set_page_config(
    page_title="Disease Prediction App", 
    layout='wide',
    page_icon="Datasets/LOGO.png"  # Provide the path to your logo file
)

# Load the pre-trained model and label encoder
with open('Pickle files/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Pickle files/label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Load the symptom mapping
with open("Pickle files/symptom_mappings.pkl", "rb") as file:
    symptom_mapping = pickle.load(file)

# Performance metrics for model explanation
accuracy = 1.0000
precision = 1.0000
recall = 1.0000
f1_score = 1.0000

# Layout: 2 columns (equal width)
col1, col2 = st.columns([1, 1])

with col1:
    # Model Explanation Section (left column)
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Disease Prediction System</h1>", unsafe_allow_html=True)

    st.subheader("Model Explanation")
    st.write("""
    This Disease Prediction System uses a machine learning model to predict diseases based on your symptoms.
    The model has been trained on a variety of diseases and their corresponding symptoms.
    The symptoms are mapped to specific features, and the model processes these features to predict the most likely disease.
    """)

    st.subheader("Performance Metrics")
    st.markdown(f"""
    - **Accuracy**: {accuracy * 100:.2f}%
    - **Precision**: {precision * 100:.2f}%
    - **Recall**: {recall * 100:.2f}%
    - **F1-Score**: {f1_score * 100:.2f}%
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing between sections

with col2:
    # User Input Section (right column)
    st.markdown("<h1 style='text-align: center;'>Enter Your Symptoms</h1>", unsafe_allow_html=True)

    # Use <h3> tag for the dropdown title
    st.markdown("<h3>Select Symptoms (Choose multiple if needed):</h3>", unsafe_allow_html=True)

    # Dropdown for symptom selection
    symptom_options = list(symptom_mapping.keys())
    selected_symptoms = st.multiselect(
        "",
        symptom_options
    )

    # Checkboxes for additional symptoms
    additional_symptoms = [
        "High Fever", "Headache", "Nausea", "Dizziness"
    ]

    additional_checkboxes = {symptom: st.checkbox(f"Have you experienced {symptom}?") for symptom in additional_symptoms}

    st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing between the checkboxes and prediction button

    # Predict Button
    if st.button("Predict"):
        if selected_symptoms or any(additional_checkboxes.values()):
            # Process user input (combine selected symptoms and checkboxes)
            symptoms = [s.strip().lower() for s in selected_symptoms]

            # Add symptoms from checkboxes if selected
            for symptom, is_checked in additional_checkboxes.items():
                if is_checked:
                    symptoms.append(symptom.lower())

            # Map symptoms to encoded values
            encoded_symptoms = []
            for symptom in symptoms:
                if symptom in symptom_mapping:
                    encoded_symptoms.append(symptom_mapping[symptom])
                else:
                    st.warning(f"Symptom '{symptom}' not recognized. Skipping...")

            # Ensure input length matches the model's requirement (10 features)
            if len(encoded_symptoms) > 10:
                encoded_symptoms = encoded_symptoms[:10]  # Truncate to first 10 symptoms
            elif len(encoded_symptoms) < 10:
                encoded_symptoms += [-1] * (10 - len(encoded_symptoms))  # Pad with -1 for missing values

            # Replace -1 with a placeholder value (e.g., mean or mode of the feature column)
            input_vector = np.array([val if val != -1 else 0 for val in encoded_symptoms]).reshape(1, -1)

            # Make prediction
            try:
                prediction = model.predict(input_vector)
                confidence_scores = model.predict_proba(input_vector)
                predicted_disease_numeric = prediction[0]  # Numeric label

                # Decode the numeric label to the disease name
                predicted_disease = le.inverse_transform([predicted_disease_numeric])[0]

                # Get the confidence score of the predicted class
                confidence_score = np.max(confidence_scores) * 100

                # Display results
                st.success(f"The predicted disease is: **{predicted_disease}**")
                st.info(f"Confidence Score: **{confidence_score:.2f}%**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter symptoms to predict the disease.")
