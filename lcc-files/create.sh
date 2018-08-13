#!/bin/bash

set -e
set -x
set -v

sudo -i -u ec2-user bash << EOF
git clone https://github.com/fastai/fastai.git /home/ec2-user/SageMaker/fastai
mkdir /home/ec2-user/SageMaker/fastai/courses/dl1/data
EOF