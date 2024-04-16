
Chinese | [English](README_en.md)

## DeepInferenceNet

## Introduction
Since transitioning from Java to Python in 2019 and after four years of accumulation, the DeepInferenceNet, a deep learning inference engine developed using the C++11 standard, finally came into being in 2022. Following features are included in the current release:

- The computation part of DeepInferenceNet does not depend on any third party, it is simple and pure; When deploying DeepInferenceNet for image tasks such as object detection, it only depends on opencv to process images. DeepInferenceNet will always maintain characteristics like few third-party dependencies and simple usage, to provide a friendly UI experience to users.
- Currently, DeepInferenceNet supports cpu inference in the x86 architecture and plans to support arm, vulkan, opengl, etc. in the future.
- DeepInferenceNet supports various operating systems such as Windows and Linux;
- DeepInferenceNet uses pure C++ code to construct a network instead of using text files. The network construction method of DeepInferenceNet is very similar to pytorch, paddlepaddle. The advantage is that you can debug the C++ code at any point of the forward propagation, as opposed to treating the deep learning inference framework as a black box. You can learn, understand and even expand it to support more operators. The downside is that you need to write C++ code manually to form a network.

Using DeepInferenceNet, PPYOLOE, PICODET and other algorithms have been implemented (as examples of using DeepInferenceNet). If you want to learn to use DeepInferenceNet, it is a good choice to look at their code. If you find this repository helpful, please give a star!

Detailed instructions for both Windows and Ubuntu are available below which should help with the setup.



## Updates!!
* [2022/11/03] DeepInferenceNet public release.


## Social Media Profiles

- AIStudio: [asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)
- WeChat: wer186259

If you liked this repo, you might consider following me on the platforms above.



## Reference

DeepInferenceNet has referred to the excellent code of the following repositories:

[Paddle](https://github.com/PaddlePaddle/Paddle)

[ncnn](https://github.com/Tencent/ncnn)

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[miemiedetection](https://github.com/miemie2013/miemiedetection)