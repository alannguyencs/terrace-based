# Terrace-based Food Counting and Segmentation
This paper represents object instance as a terrace, where the height of terrace corresponds to object attention while the evolution of layers from peak to sea level represents the complexity in drawing the finer boundary of an object. A multitask neural network is presented to learn the terrace representation. The attention of terrace is leveraged for instance counting, and the layers provide prior for easy-to-hard pathway of progressive instance segmentation. We study the model for counting and segmentation for a variety of food instances, ranging from Chinese, Japanese to Western food. This paper presents how the terrace model deals with arbitrary shape, size, obscure boundary and occlusion of instances, where other techniques are currently short of. 

Check out our conference paper [here](https://ojs.aaai.org/index.php/AAAI/article/view/16337).

The images and segmentation masks are publicly available to download [here]().

![SibNet Performance](images/result.png)

## Source code
The source code includes the implementations for:
* Terrace's architecture
* Evaluation metrics
* Data pre-processing
* Fundamental functions for training, testing, visualizing the results
* Instance extraction algorithms written in Cython

## Installation
1. Setup and activate anaconda environment:
    ```bash
    conda env update --file environment.yml --prune
    conda activate terraceenv
    ```
1. Setup cython libraries: direct to src/alcython/ and run the following bash command
    ```bash
    python setup.py build_ext --inplace
    ```
## Train and test Terrace model
The following configurations and commands works under the **src** folder
1. Config the path to data images, segmentation masks, seed masks and sibling relation maps in *constants.py*
1. How to train:
    * Edit the data type in *config.py*
    * Run ```python main.py train```
1. How to test:
    * Edit the path to trained model (ckpt_path) in *config.py*
    * Run ```python main.py test```

Our weights trained on four food datasets are found [here](https://drive.google.com/drive/folders/1IcsO6IokH-QbyeYnlcbY_Ehq8cDW9ozt?usp=sharing).

## Citation
```
@article{Nguyen_Ngo_2021, 
title={Terrace-based Food Counting and Segmentation}, 
volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/16337}, 
number={3}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Nguyen, Huu-Thanh and Ngo, Chong-Wah}, 
year={2021}, 
month={May}, 
pages={2364-2372} 
}
```

