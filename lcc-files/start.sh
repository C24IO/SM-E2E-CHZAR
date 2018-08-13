#!/bin/bash

set -e
set -x
set -v

echo "Creating fast.ai conda enviornment"
cat > /home/ec2-user/fastai-setup.sh << EOF
#!/bin/bash
cd /home/ec2-user/SageMaker/fastai
conda update -y -n base conda
conda env update
source activate fastai
conda install -y boto3
echo "Finished creating fast.ai conda environment"
EOF

chown ec2-user:ec2-user /home/ec2-user/fastai-setup.sh
chmod 755 /home/ec2-user/fastai-setup.sh

sudo -i -u ec2-user bash << EOF
echo "Creating fast.ai conda env in background process."
nohup /home/ec2-user/fastai-setup.sh &
EOF