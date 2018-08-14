
https://github.com/mdbloice/Augmentor

cd /home/ec2-user/SageMaker/fastai
conda update -y -n base conda
conda env update
source activate fastai
conda install -y boto3



aws s3 sync data.tar.gz s3://chzar/scratch/
aws s3 sync s3://chzar/scratch/data.tar.gz .

aws s3 rm s3://chzar/data --recursive


[root@ip-172-16-44-133 ~]# watch -n1 nvidia-smi

nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5

http://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf

nvidia-smi -i 0 -q -d MEMORY,UTILIZATION,ECC,TEMPERATURE,POWER,CLOCK,COMPUTE,PIDS,PERFORMANCE,SUPPORTED_CLOCKS,PAGE_RETIREMENT,ACCOUNTING

nvidia-smi -i 0 -q -d MEMORY,UTILIZATION,ECC,TEMPERATURE,POWER,CLOCK,COMPUTE,PIDS,PERFORMANCE,PAGE_RETIREMENT,ACCOUNTING


