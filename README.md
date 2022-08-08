1. Unet (unet.zip)
2. Unet++ (unet.zip)
3. Dense CNN (unet.zip)
4. ViT (vit.zip)
5. GAN (recongan.zip)


Dependencies:
Prior to executing any of the above, please install the following:

conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 h5py=3.6.0 -c pytorch

Unpack Code:

1. Unzip the vit.zip archive.
2. Unzip the recongan.zip archive.
----Unzip the above into different directories-----
3. Unzip the unet.zip archive, and move the unet_nested.py and unet_dense.py files to the fastMRI/fastmri/models directory, and move the unet_module.py to the fastMRI/fastmri/modules directory, of either the vit or recongan unzipped directories.
Install the contents of the fastMRI directories into the python environment with which you are going to run the code 
e.g. Before running ViT, run pip install -e vit/fastMRI (repeat for recongan)



Download Data:
Go to https://fastmri.med.nyu.edu/ to request access to the data. Follow instructions to download and unzip. Please see the file small_dataset_list file
for the list of files used during training for all experiments. Pass the path to which the data was downloaded in the --data argument in the commands below.


Executing code:
To execute the ViT model:
python fastmri/fastmri_examples/ViT/train_vit_demo.py --data_path path/to/data --challenge singlecoil --mask random --num_workers 0 --max_epochs 20


To execute the ReconGan model
python fastmri/train_recongan_demo.py --data_path path/to/data --challenge singlecoil --mask random --num_workers 0 --max_epochs 15
If you experience trouble, please consult the more detailed README.md file in the recongan directory.

To execute the Unet model:
python fastmri/fastmri_examples/unet/unet_demo.py --data_path path/to/data --challenge singlecoil --mask random --num_workers 0 --max_epochs 20

----To run Unet++ and Dense CNN-----
Open the fastMRI/fastmri/modules/unet_module.py file. edit the import from fastmri.models import unet and change it to either unet_nested or unet_dense depending on what you wish to execute.
After updating the import run:
python fastmri/fastmri_examples/unet/unet_demo.py --data_path path/to/data --challenge singlecoil --mask random --num_workers 0 --max_epochs 20




** Setting up A Virtual Machine on GCP**
**Setting the VM infrastructure:

Step 1: Use the coupon to get the credit in the education billing account

Step 2: Create a new project and link it to this billing account

Step 3: Submit a request to increase GPU quota to 1.

Step 4: Go to Compute Engine and create a VM instance

Step 5: Setting up the VM 
Select : 
GPU, 
number of GPU = 1, 
Machine_type = Custom, 
Memory = 20GB (click Extend memory)
Boot Disk - 100GB (select Image asked for and update the memory)
Firewall - Allow HTTP, Allow HTTPS

Step 5: The VM instance in now created. Click on SSH to connect

Step 6: Say 'Y' to install Nvidia driver

Step 7: Set up - Anaconda

wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh

Step 8: Create Static IP
VPC network - External IP - reserve

Step 9: Firewall access
VPC network - Firewall - create rule - access to port 3389


Step 10 - Update python version
conda install -c anaconda python=3.8

(check if it is updated by running python --version, if no update set up a seperate environment as per steps at the end) 



step 11: Install pytorch & dependencies
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
cd fastMRI
pip install -e .


Step 12 Configure Jupyter

source ~/.bashrc
jupyter notebook --generate-config

vi ~/.jupyter/jupyter_notebook_config.py

(Add the following line of code into the configuration file by pressing Ã¢â‚¬Å“iÃ¢â‚¬Â.)

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 3389

press ":wq"  to save and quit

Step 13 Run Jupyter notebook

jupyter-notebook --no-browser --port=3389
http://<External Static IP Address>:3389


Step14: Clone the FastMRI (start another SSH instance)

git clone https://github.com/facebookresearch/fastMRI


Step 15: Download the data (takes 6-7mins for validation set )
echo insecure >> ~/.curlrc
(put the link provided on the email received)



=========================================
ALL SET TO Run files on the Jupyter server

Step 10 & 11 (set up environment)

conda create -n newenv  python=3.8
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
cd fastMRI
pip install -e .
conda install notebook
conda install matplotlib

(move to step 12 given above)
