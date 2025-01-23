# Disease Prediction System

A machine learning-powered disease prediction system that takes user-reported symptoms and predicts the possible disease with a confidence score. The system leverages a pre-trained machine learning model to provide accurate predictions based on the provided symptoms.

## Features

- **Symptom-based Disease Prediction**: Users can input their symptoms (comma-separated or via selection) to get a disease prediction
- **Confidence Score**: The app provides a confidence score to indicate the likelihood of the predicted disease
- **Easy-to-use Interface**: A simple and interactive interface built with Streamlit, allowing users to quickly enter symptoms and view results
- **Symptom Mapping**: The system uses an encoded mapping for symptoms to predict the disease effectively

## Technologies Used

- **Python**: The core language used for the development of the app
- **Streamlit**: A framework used for building the web app interface
- **scikit-learn**: A machine learning library for loading and using the pre-trained model
- **NumPy**: A library for handling arrays and numerical operations

## Requirements

To run this app locally, ensure that you have the following installed:

- Python 3.6 or higher
- `streamlit`
- `numpy`
- `scikit-learn`
- `pickle`

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## How to Run the App Locally

Follow these steps to run the Disease Prediction System locally:

1. Clone this repository:
   ```bash
 (https://github.com/GubbalaSaiJithin/Disease-Prediction-App)
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and visit `http://localhost:8501` to interact with the app.

## App Usage

1. **Input Symptoms**: You can either type your symptoms in the input text area (comma-separated) or select from the available symptoms in the dropdown list.

2. **Click "Predict"**: After entering symptoms, click the **Predict** button to get the disease prediction along with the confidence score.

3. **Interpret Results**: The system will display the predicted disease and the confidence score, which indicates the probability of the diagnosis.

### Example:

- **Symptoms Input**: "itching, skin rash, continuous sneezing"
- **Prediction Output**: "The predicted disease is: **Skin Allergy**"
- **Confidence Score**: "Confidence Score: **95.75%**"

## Model Explanation

The model used in this app is a machine learning classifier trained on a dataset of common diseases and their associated symptoms. The model is capable of predicting a disease based on the symptoms entered by the user.

### Model Details:

- **Algorithm Used**: [XGBoost Classifier]
- **Accuracy**: The model has an accuracy of [100]% based on the evaluation on the test set
- **Input Features**: Symptoms encoded into numerical values for model prediction
- **Output**: A disease label predicted by the model, along with the confidence score

## Screenshots

![Screenshot 2025-01-07 121823](https://github.com/user-attachments/assets/cb740ce8-b00a-40d1-9494-704270f1cb01)
![Screenshot 2025-01-07 121908](https://github.com/user-attachments/assets/9147fbfc-a031-4519-b641-bc29fc301441)


## Deployment

The app is also deployed on Streamlit Cloud. You can access the live app by clicking the link below:

- [Live App Link](https://disease-prediction-app-nknheifaqume48onnplpjg.streamlit.app/)

## Author

This Disease Prediction App was developed by :
-	[@GubbalaSaiJithin](https://github.com/GubbalaSaiJithin)
-	Repository : https://github.com/GubbalaSaiJithin/Disease-Prediction-App.git

## Feedback

For any feedback or queries, please reach out to me at saijithin09@gmail.com.
