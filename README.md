# Samples visionlab_stack

> Samples for [visionlab_stack](https://github.com/visiont3lab/visionlab_stack.git) library


## Setup

* Python3.6 is required to allow compatibility with [Deepfillv2 Tensorflow](https://github.com/JiahuiYu/generative_inpainting)
    Issue python 3.8 does not support tensorflow 1.14.0

* Cloning and setup the repository 
```
git clone https://github.com/visiont3lab/visionlab_stack_samples.git
cd visionlab_stack_samples
mkdir -p data/results/detection
mkdir -p data/results/images
mkdir -p data/results/masks
mkdir -p data/results/inpainting
mkdir -p data/results/bg_estimate

virtualenv --python=python3.6 env
source env/bin/activate
pip install git+https://github.com/visiont3lab/visionlab_stack
```

* Download models.zip from this [visionlab_stack models link](https://drive.google.com/file/d/1uPLrxxxd1__WVK_xMuDPXGIWojQwBMha/view?usp=sharing).
Unzip the file and place it in **data/**

* Run people remove

```
python people_remove.py
```

## Credits to

* [Yolov5 Pytorch](https://github.com/ultralytics/yolov5)
* [Yolov3/v4 Darknet](https://github.com/AlexeyAB/darknet)
* [Deepfillv2 Tensorflow](https://github.com/JiahuiYu/generative_inpainting)
* [Deepfillv1 Pytorch](https://github.com/vt-vl-lab/FGVC)
