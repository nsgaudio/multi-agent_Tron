### Azure

> [http://web.stanford.edu/class/cs234/assignment2/CS234%20Azure%20Setup.pdf](http://web.stanford.edu/class/cs234/assignment2/CS234 Azure Setup.pdf)



```shell
# part 1
git clone https://github.com/AndyYSWoo/Azure-GPU-Setup.git
cd Azure-GPU-Setup
chmod +x *.sh
./gpu-setup-part1.sh

# part 2
nvidia-smi
cd Azure-GPU-Setup

####### don't install their requirement ######
rm requirements.txt
touch requirements.txt
####### don't install their requirement ######

./gpu-setup-part2.sh
python gpu-test.py

# part 3

###### copy paste #########
sudo apt-get install python3.6-tk
python3.6 -m pip install --upgrade
python3.6 -m pip install -U numpy
python3.6 -m pip install matplotlib==2.0.2
python3.6 -m pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
###### copy paste #########


```





### AWS

> <http://cs231n.github.io/aws-tutorial/>



### GCP

> <https://github.com/cs231n/gcloud>



