
# Introduction
This repository contains the source code, Jupyter notebooks, configuration etc. for an End-2-End Model training to deployment SageMaker demo

Source: https://github.com/mattmcclean/sagemaker-lhr-summit-demo

Workshop 1 must be in US East (N. Virginia)	us-east-1
======

Setup
------

1. Launch a new Cloud9 instance. Make sure you have enough space on the EBS volume. *Advanced Step* - You may have to stop the Cloud9 instance and resize the EBS volume as the default size is only 8 GB.

2. Copy the repo - `git clone https://github.com/C24IO/SM-E2E-CHZAR.git`

3. Goto AWS S3 Console & create an S3 bucket for the model artefacts. The S3 bucket should be named: ```sagemaker-<region_name>-<account_id>``` where *<region_name>* is the AWS region where the demo is being run and *<account_id>* is the AWS Account ID. Please create this in ```US East (N. Virginia)	us-east-1```. This step is really important for the workshop to work properly.

    If you need to create the S3 bucket do so with this AWS CLI command:

    ```aws s3 mb s3://sagemaker-$(aws configure get region)-$(aws sts get-caller-identity --query 'Account' --output text) --region us-east-1 ```

You can run the aws commands right in the Cloud9 Terminal

4. We will now prepare to install custom Conda Kernel base on FastAI Library. Goto AWS SageMaker Console & - please create a new LifeCycle Configuration. Goto "Lifecycle configurations" in the SageMaker Dashboard. And "Create configuration".

5. Now lets create an instance to use those Lifecycle configurations - Everything else should be left to default.  *Advanced Step* - Select Accelerated Computing Notebook instance type and do training in your own notebook. Instead of using a pre-trained model for our endpoint (coming later in the workshop).
 


6. In Start notebook copy and paste script from - `SM-E2E-CHZAR/lcc-files/start.sh`

7. In Create notebook copy and paste script from - `SM-E2E-CHZAR/lcc-files/create.sh`. *Advanced Step* - Try to play around with these scripts. 



Lifecycle configurations - put them in 

4. Create SageMaker Notebook 

Open terminal of Sagemaker


git clone https://github.com/C24IO/SM-E2E-CHZAR.git


source activate fastai
conda install -y boto3

(fastai) sh-4.2$ pwd
/home/ec2-user/SageMaker/SM-E2E-CHZAR
(fastai) sh-4.2$ mv ../fastai/fastai .
(fastai) sh-4.2$


Get the data - 


mkdir data
cd data
http://files.fast.ai/data/dogscats.zip
wget -cv http://files.fast.ai/data/dogscats.zip
unzip dogscats.zip




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


Steps
------

1. 


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




