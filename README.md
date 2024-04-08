# AI Integration for Company Products

This repository contains a Python script and datasets used in a hackathon project aimed at integrating AI into our company's products. The project demonstrates a practical application of machine learning to predict customer behavior based on bank account details, leveraging AWS SageMaker for model training and deployment.

## Dependencies

Before running the script, ensure you have the following dependencies installed:

- Python 3.x
- pandas
- scikit-learn
- Boto3 (AWS SDK for Python)
- AWS SageMaker

Install these dependencies using pip:

```bash
pip install pandas scikit-learn boto3 sagemaker
```

## Datasets

The project utilizes several datasets for training and validation:

- `AWSSagemaker.csv`: Main dataset with customer details and transactional data.
- `train_features.csv` & `train_labels.csv`: Split from the main dataset for training, where `train_features.csv` contains the features and `train_labels.csv` contains the target variable.
- `validation.csv`: Used for model validation, containing both features and the target variable.
- `train.csv`: An additional training dataset provided for context.
- `bank_accounts_details_categorized.xlsx`: Contains categorized bank account details for further analysis or model training.

## How to Run

Ensure you have configured your AWS credentials and SageMaker permissions. Then, execute the script:

```bash
python BankAccountDetails.py
```

The script will preprocess the data, train a model using AWS SageMaker, and deploy the model to an endpoint for predictions.

## Output

The script outputs:

- CSV files for training and validation data, ready for upload to AWS SageMaker.
- A deployed SageMaker endpoint for real-time predictions.
- A sample AWS Lambda function code for invoking the SageMaker endpoint.

For further details on interpreting the outputs and integrating them into company products, refer to the script comments and AWS SageMaker documentation.
