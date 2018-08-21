
# Introduction
This repository contains the source code, Jupyter notebooks, configuration etc. for an End-2-End Model training to deployment SageMaker demo

Source: https://github.com/mattmcclean/sagemaker-lhr-summit-demo

## Workshop 1 must be in US East (N. Virginia)	us-east-1

### Part 1 - Setup notebook and train the model
-----------

1. Launch a new Cloud9 instance. Make sure you have enough space on the EBS volume. *Advanced Step* - You may have to stop the Cloud9 instance and resize the EBS volume as the default size is only 8 GB. This usecase would require us to increase the size of EC2 instance that is hosting our Cloud9 instance. For this please navigate to the AWS EC2 console, and find the EC2 instance that is hosting our Cloud9 instance.
    1. Select the instance that has the name of our Cloud9 instance. 
    2. Select the Root device and navigate to the volume thats attached to this disk
    3. Once the EBS volume is selected we go to Actions and select Modify Volume and change the size to 32 GB and click Modify
    4. Wait for the volume size to reflect the new size of 32GB, then reboot the EC2 instance. 
    5. Go to the terminal of Cloud9 instance and see if the size is now 32GB.

2. Copy the repo - `git clone https://github.com/C24IO/SM-E2E-CHZAR.git`

3. Goto AWS S3 Console & create an S3 bucket for the model artifacts. The S3 bucket should be named: ```sagemaker-<region_name>-<account_id>``` where *<region_name>* is the AWS region where the demo is being run and *<account_id>* is the AWS Account ID. Please create this in ```US East (N. Virginia)	us-east-1```. This step is really important for the workshop to work properly.

    If you need to create the S3 bucket do so with this AWS CLI command:

    ```aws s3 mb s3://sagemaker-$(aws configure get region)-$(aws sts get-caller-identity --query 'Account' --output text) --region us-east-1 ```

    You can run the aws commands right in the Cloud9 Terminal

4. We will now prepare to install custom Conda Kernel base on FastAI Library. Goto AWS SageMaker Console & - please create a new LifeCycle Configuration. Goto "Lifecycle configurations" in the SageMaker Dashboard. And "Create configuration". Name it something like `FastAI`

5. In Start notebook copy and paste script from - `SM-E2E-CHZAR/lcc-files/start.sh`

6. In Create notebook copy and paste script from - `SM-E2E-CHZAR/lcc-files/create.sh`. *Advanced Step* - Try to play around with these scripts. 

7. Now lets create an instance to use those Lifecycle configurations -   
    1. **Notebook instance name** - `SME2E-<username>`
    2. **Notebook instance type** - `ml.m4.xlarge` - Any instance type would do, as we are not going to train the model in this notebook today. *Advanced Step* - Select Accelerated Computing Notebook instance type and do training in your own notebook. Instead of using a pre-trained model for our endpoint (coming later in the workshop)
    3. **IAM role** - `Create a new role` - Please select `Any S3 bucket`. *Advanced Step* - You can edit this role from the IAM console later on
    4. **VPC - optional** - No VPC
    5. **Lifecycle configuration - optional** - `FastAI` - The one that you created earlier
    6. **Encryption key - optional** - `No Custom Encryption`

8. Now please wait while the instance launches. And get ready to clone the github repo into this SageMaker Instance. *Advanced Step* - Enable logging in the Lifecycle configurations by editing the scripts to log to stdout

9. Now once the SageMaker instance has launched - Goto Dashboard->Notebook instances->`SME2E-<username>`->Actions->Open. You would see a Jupyter notebook - now navigate to over the right hand side of that notebook and open `New->Terminal`. You will be presented with a Web-based Linux terminal. 
    1. In terminal navigate to - `cd /home/ec2-user/SageMaker` 
    2. Copy the repo again - `git clone https://github.com/C24IO/SM-E2E-CHZAR.git`
    3. We are going to get data for this notebook to work on - Go into the repo that we cloned `cd SM-E2E-CHZAR/`
    4. `mkdir data`
    5. `cd data`
    6. `wget -cv http://files.fast.ai/data/dogscats.zip`
    7. `unzip dogscats.zip` - now you have the data required to run this notebook in `SM-E2E-CHZAR/data/dogscats` - we use this location in the next steps
    3. Navigate back to the Jupyter notebook and descend into - `SM-E2E-CHZAR`
    4. Open the prebuilt notebook there - `build_train_custom_model.ipynb`
    5. This notebook contains bare-bones code for *Image classification with Convolutional Neural Networks* for the full notebook with comments, helpful code its hosted [here](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb).
    6. The lesson that you can follow to go in the details of this code to *RECOGNIZING CATS AND DOGS* is [here](http://course.fast.ai/lessons/lesson1.html)

#### Optional - *Advanced Steps* - To train your own model (40+ mins)

[Normal steps resume later in the workshop here](#normal-workshop-resumes)

10. Edit the notebook cell to make this notebook your very own - 
```python
bucket='sagemaker-chazarey-us-east-1'   # customize to the name of your S3 bucket
model_file_name = 'sm-e2e-chzar-model'  # customize to the name of your model
PATH='data/dogscats/'                   # customize to the relative location of your data folder
key='models/'+model_file_name+'/model.tar.gz' # prefix of the S3 bucket of the model file
```

11. We might run into issues while importing boto3 library
```python
import boto3
```
To fix this please do the following on the Jupyter notebook terminal - 
```bash
cd /home/ec2-user/SageMaker/fastai
source activate fastai
conda install -y boto3
```

12. Lets make the fastai library local to this notebook so we can 
```python
import torch
from fastai.imports import *
```
To fix this please do the following on the Jupyter notebook terminal - 
```bash
cd /home/ec2-user/SageMaker/SM-E2E-CHZAR
cp -aprvf /home/ec2-user/SageMaker/fastai/fastai .
```

13. Please go ahead from here and train the model, this notebook will also upload your model artifacts into S3 bucket you specified before.

#### Normal Workshop resumes
#normal-workshop-resumes

### Part 2 - Create a model and deploy an endpoint
-----------

##### Let's take a look at how we would use a pretrained model and deploy an endpoint using that

1. Navigate to your Cloud9 instance and go into the terminal in that instance

2. To create a SageMaker Model We would need to have a container to run inference code against our model artifacts. Towards that end, we have a preconfigured container image template in the github repo - SM-E2E-CHZAR/container. Please navigate to SM-E2E-CHZAR/container. and execute ```./build_and_push.sh sm-e2e-chzar-container ``` This step would take about 10 mins or so to complete. 

3. Details about the engineering of this docker image are dealt with more details in this [sample notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb) and a [published blog](https://aws.amazon.com/blogs/machine-learning/train-and-host-scikit-learn-models-in-amazon-sagemaker-by-building-a-scikit-docker-container/).

4. If you note the steps that are used to create this endpoint - we use the [CPU version](https://github.com/fastai/fastai/blob/master/environment-cpu.yml) of files needed as compared to [GPU version](https://github.com/fastai/fastai/blob/master/environment.yml) that we used for training. 

5. Once this step is complete, you should navigate to AWS ECS console and checkout the container that we just pushed. Now copy the Repository URI which looks like this - `111652037296.dkr.ecr.us-east-1.amazonaws.com/sm-e2e-chzar-container`. We use this later in model creation.

6. Before we move on to the actual SageMaker model creation, lets upload our model artifacts into S3. Do this on your Cloud9 Terminal

    1. Navigate into the cloned github repo - `SM-E2E-CHZAR/model` 
    2. Here you can find a pretrained model - `model.tar.gz` 
    3. Let's upload it to our S3 bucket we created earlier - `aws s3 cp model.tar.gz  s3://sagemaker-1191-us-east-1-111652037296/`
    4. Here is the location for our model artifacts now - `s3://sagemaker-1191-us-east-1-111652037296/model.tar.gz`
 
6. Now we move on to Model Creation in SageMaker - 
    1. Navigate to Dashboard->Inference->Models->Click `Create model`
    2. **Model name** - `sm-2e2-chzar-model`
    3. **IAM role** should be already populated - as it was created for your notebook instance before
    4. **Network** - leave at the default setting
    5. **Primary container** - Let's create the model serving endpoint container now
    6. **Location of inference code image** - the URI from our previous container deployment - `111652037296.dkr.ecr.us-east-1.amazonaws.com/sm-e2e-chzar:latest`
    7. **Location of model artifacts** - optional - `s3://sagemaker-1191-us-east-1-111652037296/model.tar.gz`
    8. Finish the process by clicking on `Create model`

7. Now we at the step just before we deploy an endpoint - we create an endpoint configuration
    1. Only 2 things needed to be filled in here - *Endpoint configuration name* - `sm-e2e-chzar-endpoint-config`
    2. Click on **Add model** & add the model that we created in the previous step
    3. Click on `Create endpoint configuration`

8. Finally - the last step before our endpoint is in production - 
    1. Goto Dashboard->Endpoints->Click on `Create endpoint`
    2. **Endpoint name** - `sm-e2e-chzar-endpoint`
    3. In **Endpoint configuration** - select the configuration that we created in the step above
    4. By default we select `ml.m4.xlarge`, please note this is a non GPU instance, as we are performing inference only and not learning. 

9. Now finally we have a deployed endpoint - named `sm-e2e-chzar-endpoint`. 


10. Setup for running inference - Please execute this in your Cloud9 Terminal
```bash

sudo pip install boto3
cd /home/ec2-user/environment/SM-E2E-CHZAR
mkdir data
cd data


chzar:~/environment/SM-E2E-CHZAR (master) $ ./predict_img_endpoint.py -e sm-2e2-chzar-model-2-endpoint
Traceback (most recent call last):
  File "./predict_img_endpoint.py", line 9, in <module>
    import boto3
ImportError: No module named boto3

```

10. At the end - lets run inference against this endpoint and check our results
    1. Please navigate to your Cloud9 instance
    2. Get to the place where we cloned our git repo in the beginning
    3. There you would find a file - `predict_img_endpoint.py` - we will use this to run inference against our deployed endpoint. 
    4. Please execute this command to run an inference per second against the endpoint - `watch -n1 ./predict_img_endpoint.py -e sm-e2e-chzar-endpoint`
    5. Finally - lets go to the Endpoint and open `View invocation metrics` `View instance metrics` & `View logs` in three diffrent tabs. 
    8. Please observe the diffrent metrics and see how the deployed endpoint reacts to variable loads.


### Optional - *Advanced Steps* - Model A/B testing     
    
1. We can all run inference against one endpoint in the class and see how the infrastructure performs.
2. We can use an inferior instance type to create the endpoint and see how the infrastructure treats failure and if we can create mechanism to handle that.
3. [You can perform A/B testing here by creating multiple endpoints as detailed in this blog](https://medium.com/@julsimon/mastering-the-mystical-art-of-model-deployment-c0cafe011175). You can create diffrent version of the model by tuning the hyperparameters to diffrent values. 

### Optional - *Advanced Steps* - Extend notebook to adapt to diffrent images

Please note the structure of our directory here - 

```bash
.
└── dogscats
    ├── models
    ├── sample
    │   ├── train
    │   │   ├── cats
    │   │   └── dogs
    │   └── valid
    │       ├── cats
    │       └── dogs
    ├── test1a
    ├── train
    │   ├── cats
    │   └── dogs
    └── valid
        ├── cats
        └── dogs
```


You can adapt this structure, change names of directories and use this same notebook and the same model to [create a world class image classifier for *any* types of images](https://medium.com/@apiltamang/case-study-a-world-class-image-classifier-for-dogs-and-cats-err-anything-9cf39ee4690e). 

Links
------
* Building, Training and Deploying Custom Algorithms Such as Fast.ai with Amazon SageMaker
https://youtu.be/1kJf0Lvzj8A

* LifeCycle Configuration logs 
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logStream:group=/aws/sagemaker/NotebookInstances;prefix=SageMaker-E2E-Try3;streamFilter=typeLogStreamPrefix

* LifeCycle Config - 
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instance-lifecycles/FastAIConfig

* Ready Notebook Running
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances/SageMaker-E2E-Try3

* Lesson 1: Deep Learning 2018
https://www.youtube.com/watch?v=IPBSB1HLNLo

* Model files
 https://s3.amazonaws.com/sagemaker-e2e-try3.notebook.us-east-1.sagemaker.aws/models/lhr-summit-demo/model.tar.gz 
 + s3://sagemaker-e2e-try3.notebook.us-east-1.sagemaker.aws/models/lhr-summit-demo/model.tar.gz
 + 