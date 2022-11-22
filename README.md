# DynaPP

Our code is based on https://github.com/ultralytics/yolov5.

## Please read 'guideline.pdf'

---

Common options ("val.py") (will be updated)

"--": 


---

## Prepare hardware

Our code is based on PyTorch.

Installing PyTorch in jetson edge devices is different from general installation.

Jetson Nano : https://qengineering.eu/install-pytorch-on-jetson-nano.html

Nvidia Jetson TX2 : https://medium.com/hackers-terminal/installing-pytorch-torchvision-on-nvidias-jetson-tx2-81591d03ce32

Wheel installers : https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

+ If error occurs as : usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static tls bloc

Write this before access jupyter notebook : export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

---

## Please download weight files below

### (Put the files in 'weights' folder)

https://drive.google.com/file/d/1LTSKE19bpygugylP9jMk2dtjdgcQZ1vu/view?usp=share_link

https://drive.google.com/file/d/19zIMTZzF9tqOnpDBxMkoKz6u7S3-x7CW/view?usp=share_link

---

## Please download datasets below 
### (Put the files in directory you want, and modify the code inside 'Run.ipynb'

These dataset annotations have been converted to yolo format.

##### AUAIR

https://drive.google.com/file/d/1syHeOWTO5cIw3pjE68TWQdhzZPfTsHTv/view?usp=share_link

##### VisDrone

https://drive.google.com/file/d/1f02BSNxu0QAkimABYEJeLMSR01Tk1Tnr/view?usp=share_link

##### UAVDT

https://drive.google.com/file/d/1MpPPzEgjuRH3DjwFE0jhDxscSzqMjPpW/view?usp=share_link

##### ImageVID

https://drive.google.com/file/d/1w_K7uV4C_VxM5NryFpJFQC8OtSZbPIde/view?usp=share_link

---
## Hardware test
### We provide tested results on Jetson Nanon and TX2 : 'Time measured by hardwares.xlsx'

![image](https://user-images.githubusercontent.com/118588373/203310683-7ea2fe02-b5e8-4d04-96ab-f1a6b3f26947.png)
![not_square_Nano](https://user-images.githubusercontent.com/118588373/203301518-3fcd6475-a6cc-402f-a62b-2725244eea48.png)


![TX2](https://user-images.githubusercontent.com/118588373/203301579-1c667e59-f192-412b-b3ed-a3af36c5a9a6.png)
![not_square_TX2](https://user-images.githubusercontent.com/118588373/203301641-c4e99322-ad9c-4d66-a1e1-bb2e640a8f01.png)


Depending on the hardware, the inference time may not be accelerated as much as the operation costs reduction.

So please check your environment with the code: 'Test_your_hardware.ipynb'

We strongly recommend using Jetson Nvidia TX2 and Jetson Nano or with hardware of a similar specification.

However, if there is none, please experiment with existing hardware and refer to the result of acceleration indirectly with *average resolution* in 'excel_results/files'.

