
# Surgery Outcome Prediction

This project is a Streamlit application for predicting surgery outcomes based on doctor's notes and patient details. It uses machine learning models to classify surgery types and predict the outcome based on various features.

## Features

- **Predict Surgery Types**: Classify the type of surgery based on doctor's notes.
- **Predict Surgery Outcome**: Predict whether a surgery will pass or fail based on patient details and predicted surgery types.

## Models Used

- **BiLSTM Model**: For predicting surgery types from doctor's notes.
- **Gradient Boosting Model**: For predicting the surgery outcome.

## Libraries

This project uses the following Python libraries:

- `pandas`
- `numpy`
- `joblib`
- `gensim`
- `tensorflow`
- `streamlit`
- `scikit-learn`

## How to Run

1. **Clone the Repository**:
   ```shell
   git clone https://github.com/mishra-krishna/Surgery-Outcome-Prediction
   ```

2. **Install Dependencies**:
   Install the required libraries using the `requirements.txt` file.
   ```shell
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Start the Streamlit app with:
   ```shell
   streamlit run main.py
   ```

## Deployed App

You can access the deployed Streamlit app [here](https://surgery-outcome-prediction-shnrd2lzvyq8f5qn8u8xyn.streamlit.app/).

## Model Development

For details on the model development process and dataset insights, please refer to the `model_development.ipynb` notebook.

## Model Files

Ensure that the following files are present in the project directory for the app to work:

- `bilstm_model.h5` - BiLSTM model for surgery type prediction.
- `gradient_boosting_model.pkl` - Gradient Boosting model for outcome prediction.
- `word2vec_model.bin` - Word2Vec model for text processing.

## Contributing

Feel free to open issues or submit pull requests for improvements.

