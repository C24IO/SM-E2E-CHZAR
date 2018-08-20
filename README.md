
# Introduction
This repository contains the source code, Jupyter notebooks, configuration etc. for an End-2-End Model training to deployment SageMaker demo

Source: https://github.com/mattmcclean/sagemaker-lhr-summit-demo

### Workshop 1 must be in US East (N. Virginia)	us-east-1

-----------

1. Launch a new Cloud9 instance. Make sure you have enough space on the EBS volume. *Advanced Step* - You may have to stop the Cloud9 instance and resize the EBS volume as the default size is only 8 GB.

2. Copy the repo - `git clone https://github.com/C24IO/SM-E2E-CHZAR.git`

3. Goto AWS S3 Console & create an S3 bucket for the model artefacts. The S3 bucket should be named: ```sagemaker-<region_name>-<account_id>``` where *<region_name>* is the AWS region where the demo is being run and *<account_id>* is the AWS Account ID. Please create this in ```US East (N. Virginia)	us-east-1```. This step is really important for the workshop to work properly.

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

[Normal steps resume later in the workshop here](#Normal-Workshop-resumes)

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
source activate fastai
conda install -y boto3
```

#### Normal Workshop resumes 

#### Optional - *Advanced Steps* - Extend notebook to adapt to diffrent images

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
    ├── test1
    ├── train
    │   ├── cats
    │   └── dogs
    └── valid
        ├── cats
        └── dogs
```


5. Swich over to the notebook - and walk through that - people can execute if they want to execure they can otherwise its just moving on to the model downloads 
6. 

Download model 

7. Switch over to Cloud 9 and walk through the code 
8. Then deplpy the docker 
9. Check docker at ECS 
10. Copy over the URI
11. Create the model configuration
12. Create endpint configuration
13. create endpoints 
14. run inerences gainst personal
15. run inferences against demo persons
16. check the cloudwartch logs and metrics 
17. Now talk about the architecutre of inference endpoit 
18. Then show - https://github.com/aws/sagemaker-containers
19 watch -n1  ./predict_img_endpoint.py -e sm-e2e-chzar-endpoint - go to sagemaker terminal and execute this to check the status - see if you can do 3 


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




