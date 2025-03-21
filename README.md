# Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques. The dataset used is the Telco Customer Churn dataset.

## Repository Contents

- **model.ipynb**: Jupyter Notebook containing data analysis, preprocessing, model training, and evaluation.
- **WA_Fn-UseC_-Telco-Customer-Churn.csv**: Dataset used for training and evaluation.
- **Customer_Churn_model.pkl**: Serialized trained model.
- **encoders.pkl**: Serialized encoders for categorical variables.
- **requirements.txt**: List of Python dependencies required to run the notebook and scripts.

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sameer-at-git/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install dependencies**:

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use 'env\Scripts\activate'
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:

   Launch Jupyter Notebook to explore and execute the code.

   ```bash
   jupyter notebook model.ipynb
   ```

## Usage

### Load the trained model and encoders:

```python
import pickle

# Load the trained model
with open("Customer_Churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the encoders
with open("encoders.pkl", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)
```

### Make predictions on new data:

```python
import pandas as pd

# Example new customer data (ensure it follows the same preprocessing as training data)
new_data = pd.DataFrame({
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [12],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [70.0],
    "TotalCharges": [830.5]
})

# Apply encoding transformations
for col in encoders:
    new_data[col] = encoders[col].transform(new_data[col])

# Make prediction
prediction = model.predict(new_data)
print("Churn Prediction:", prediction)
```

## Notes

- Ensure that the dataset file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is present in the repository directory.
- The serialized model (`Customer_Churn_model.pkl`) and encoders (`encoders.pkl`) can be loaded to make predictions without retraining.
- For any issues or contributions, please refer to the repository's issue tracker.
