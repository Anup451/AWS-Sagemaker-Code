#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('AWSSagemaker.csv')

# Splitting the dataset into features and the target variable

X = data.drop(['customer_goodness'], axis=1)
y = data['customer_goodness']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Saving to CSV without headers and index for SageMaker
X_train.to_csv('train_features.csv', header=False, index=False)
y_train.to_csv('train_labels.csv', header=False, index=False)
pd.concat([y_val, X_val], axis=1).to_csv('validation.csv', header=False, index=False)


# In[3]:


data.describe()


# In[6]:


data.drop('age', axis=1, inplace=True) 


# In[7]:


data.describe()


# In[8]:


data.drop('city_code', axis=1, inplace=True) 


# In[9]:


import sagemaker

session = sagemaker.Session()
bucket = session.default_bucket()  # or specify your custom bucket name
prefix = 'sagemaker/your-model-name'

# Upload the data
input_train = session.upload_data('train_features.csv', bucket=bucket, key_prefix=prefix+'/train')
input_train_labels = session.upload_data('train_labels.csv', bucket=bucket, key_prefix=prefix+'/train')
input_validation = session.upload_data('validation.csv', bucket=bucket, key_prefix=prefix+'/validation')


# In[10]:


from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

role = get_execution_role()
container = get_image_uri(session.boto_region_name, 'xgboost', '1.0-1')

xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    instance_count=1,
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=session)

xgb.set_hyperparameters(objective='reg:squarederror',  # Use 'binary:logistic' for classification
                        num_round=100)


# In[21]:


import os
import sagemaker
import pandas as pd
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import image_uris
from sklearn.model_selection import train_test_split

# Define AWS SageMaker session & role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Load dataset
data = pd.read_csv('AWSSagemaker.csv')

# Assuming 'customer_goodness' is your target variable
X = data.drop(['customer_goodness'], axis=1)
y = data['customer_goodness']

X.drop('age', axis=1, inplace=True) 
X.drop('city_code', axis=1, inplace=True) 

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# SageMaker expects no header in the CSV files and the target variable in the first column
train_data = pd.concat([y_train, X_train], axis=1)
val_data = pd.concat([y_val, X_val], axis=1)

train_data.to_csv('train.csv', header=False, index=False)
val_data.to_csv('validation.csv', header=False, index=False)

# Upload the files to S3
s3_train_data = sagemaker_session.upload_data(path='train.csv', key_prefix='data')
s3_validation_data = sagemaker_session.upload_data(path='validation.csv', key_prefix='data')

# Define input data format for SageMaker
s3_input_train = sagemaker.inputs.TrainingInput(s3_data=s3_train_data, content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=s3_validation_data, content_type='csv')


X.describe()


# In[22]:


# Get the XGBoost container image
xgboost_container = image_uris.retrieve('xgboost', sagemaker_session.boto_region_name, version='1.0-1')

# Initialize the estimator
xgboost = sagemaker.estimator.Estimator(image_uri=xgboost_container,
                                        role=role,
                                        instance_count=1,
                                        instance_type='ml.m5.large',
                                        output_path='s3://{}/output'.format(sagemaker_session.default_bucket()),
                                        sagemaker_session=sagemaker_session)

# Set hyperparameters (adjust these as needed)
xgboost.set_hyperparameters(objective='reg:squarederror',
                            num_round=100,
                            max_depth=5,
                            eta=0.2,
                            gamma=4,
                            min_child_weight=6,
                            subsample=0.8,
                            silent=0)

# Train the model
xgboost.fit({'train': s3_input_train, 'validation': s3_input_validation})


# In[23]:


# Deploy the trained model to create a predictor
xgboost_predictor = xgboost.deploy(initial_instance_count=1, instance_type='ml.m5.large')


# In[24]:


from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Setup the serializer and deserializer
xgboost_predictor.serializer = CSVSerializer()
xgboost_predictor.deserializer = JSONDeserializer()


# In[25]:


# Prepare a sample data row from the validation set for prediction (excluding the target variable)
sample_data = X_val.iloc[0].values

# Convert the array to a CSV string
payload = ','.join(map(str, sample_data))


# In[26]:


# # Making the prediction
# result = xgboost_predictor.predict(payload)

# print("Prediction result:", result)


# In[31]:


# Example: Encoding categorical features before prediction (this is just illustrative)
# Assuming 'X_val' is your dataframe and you need to encode 'category_feature'

# NOTE: Replace this with your actual preprocessing steps
# For demonstration, let's assume we convert a categorical feature to numeric manually

X_val_processed = X_val.copy()

# X_val_processed['encoded_feature'] = X_val['category_feature'].apply(lambda x: 1 if x == 'SomeValue' else 0)

# Prepare the sample data row for prediction (excluding any non-numeric or target columns)
sample_data_processed = X_val_processed.iloc[0].values
payload = ','.join(map(str, sample_data_processed))

# Make the prediction with the processed payload
result = xgboost_predictor.predict(payload)
print("Prediction result:", result)


# In[32]:


import pandas as pd
from sagemaker import get_execution_role

# Load the data
file_path = 'bank_accounts_details_categorized.xlsx'  # Update this path
data = pd.read_excel(file_path)

# Display the first few rows to understand the data
print(data.head())


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming there are no missing values or they have been handled

# Encode the 'category' column
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Split the data into features and target variable
X = data.drop(['category', 'category_encoded'], axis=1)
y = data['category_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[34]:


import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize SageMaker session and get execution role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Get the default S3 bucket
bucket_name = sagemaker_session.default_bucket()
# prefix = 'sagemaker/your-project-name'  
prefix = 'sagemaker/404-Hack'  

# Path to save the CSV files locally before uploading
train_path = 'train.csv'
validation_path = 'validation.csv'

# Save the train and validation data to CSV files
pd.concat([y_train, X_train], axis=1).to_csv(train_path, index=False, header=False)
pd.concat([y_test, X_test], axis=1).to_csv(validation_path, index=False, header=False)

# Upload the dataset to S3
train_input = sagemaker_session.upload_data(path=train_path, bucket=bucket_name, key_prefix=f'{prefix}/train')
validation_input = sagemaker_session.upload_data(path=validation_path, bucket=bucket_name, key_prefix=f'{prefix}/validation')

# Specify the XGBoost model
region = boto3.Session().region_name
xgboost_container = sagemaker.image_uris.retrieve('xgboost', region, version='1.0-1')

# Initialize the estimator
xgboost = sagemaker.estimator.Estimator(image_uri=xgboost_container,
                                        role=role,
                                        instance_count=1,
                                        instance_type='ml.m5.xlarge',
                                        output_path=f's3://{bucket_name}/{prefix}/output',
                                        sagemaker_session=sagemaker_session)

# Set the hyperparameters
xgboost.set_hyperparameters(max_depth=5,
                            eta=0.2,
                            gamma=4,
                            min_child_weight=6,
                            subsample=0.8,
                            silent=0,
                            objective='multi:softmax',
                            num_class=3, 
                            num_round=100)


# In[54]:


print(prefix)


# In[35]:


# Specify the input data channels
data_channels = {
    'train': TrainingInput(train_input, content_type='csv'),
    'validation': TrainingInput(validation_input, content_type='csv')
}

# Train the model
xgboost.fit(data_channels)

# Deploy the model
predictor = xgboost.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')

# Specify serializer and deserializer for the predictor
predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()


# In[36]:


# Retrieve the endpoint name
endpoint_name = predictor.endpoint_name

print(f"Created endpoint: {endpoint_name}")


# In[38]:


import boto3
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Initialize the boto3 client for SageMaker Runtime
runtime = boto3.client('sagemaker-runtime')


# In[39]:


# Assuming X_test and y_test are your features and labels for the test dataset, and endpoint_name is your endpoint's name

# Prepare the test data (without labels) for prediction
# Convert the dataframe to CSV string
test_data = X_test.to_csv(header=False, index=False)

# Use the endpoint to make predictions
response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType='text/csv',
                                   Body=test_data)

# Parse the response to get predictions
result = response['Body'].read().decode('utf-8')
predictions = np.fromstring(result, sep=',')

# Since predictions might be probabilities, you may need to convert them to your label format depending on your model's output
# For a classification task, you might round them or take the argmax if multiple classes' probabilities are returned

# Example for binary classification (modify as needed):
predicted_labels = np.round(predictions).astype(int)

# Compute various evaluation metrics
print("Accuracy:", accuracy_score(y_test, predicted_labels))
print("\nClassification Report:\n", classification_report(y_test, predicted_labels))


# In[53]:


import boto3

# Initialize the boto3 client for SageMaker Runtime
runtime = boto3.client('sagemaker-runtime')

# Sample data to be sent for prediction
# data = "15703381.04,15538627.56,641648.22,7352,0,2027164,7109385,39611040,43,22"  # Your data here, formatted as a CSV string

data = "10000000000,10000000000,10000000000,10000000000,10000000000,10000000000,10000000000,10000000000,10000000000,10000000000" 
data = "0,0,0,0,0,0,0,0,0,0" 

# Send data to the endpoint for prediction
response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType='text/csv',
                                   Body=data)

# Parse the response
prediction = response['Body'].read().decode('utf-8')

print(f"Prediction: {prediction}")


# In[51]:


import os
import boto3
import json

# Define the SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Define the handler function that the Lambda service will use as an entry point
def lambda_handler(event, context):
    # Extract the payload from the event and format it as CSV
    body = json.loads(event['body'])
    data = ','.join(map(str, body['data']))

    # Specify your SageMaker endpoint name
#     endpoint_name = os.environ['ENDPOINT_NAME']
    endpoint_name='sagemaker-xgboost-2024-03-15-07-10-39-026'
    
    # Use the SageMaker runtime to invoke the endpoint
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType='text/csv',
                                       Body=data)
    
    # Decode the response
    result = response['Body'].read().decode('utf-8')
    print("result + ", result)
    # You may want to process the result further depending on your model's output format
    # For simplicity, we'll return the raw result here
    return {
        'headers': {
            'Content-Type': 'application/json'
        },
        'statusCode': 200,
        'endpoint':endpoint_name,
        'body': json.dumps({'result': result,
            'statusCode': 200,
            'endpoint' : endpoint_name
        })
    }

# The following lines are used for local testing of the lambda_handler function
# They simulate an API Gateway event
if __name__ == '__main__':
    fake_event = {
        'body': json.dumps({
            'data': [15703381.04, 15538627.56, 641648.22, 7352, 0, 2027164, 7109385, 39611040, 43, 22]
        })
    }
    fake_context = {}
    print(lambda_handler(fake_event, fake_context))


# In[ ]:




