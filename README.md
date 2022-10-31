# TO-Scene: A Large-scale Dataset for Understanding 3D Tabletop Scenes
<img src="./figure/pipeline_new.jpg" width="900"/>

By [Mutian Xu*](https://mutianxu.github.io/), [Pei Chen*](), [Haolin Liu](), and [Xiaoguang Han](https://gaplab.cuhk.edu.cn/)

## Introduction
This repository is built for:

__TO-Scene__: A Large-scale Dataset for Understanding 3D Tabletop Scenes ___(ECCV2022 Oral)___ [[arXiv](https://arxiv.org/abs/2203.09440)]
<br>


If you find our work useful in your research, please consider citing:

```
@inproceedings{xu2022toscene,
  title={TO-Scene: A Large-scale Dataset for Understanding 3D Tabletop Scenes},
  author={Xu, Mutian and Chen, Pei and Liu, Haolin and Han, Xiaoguang},
  booktitle={ECCV},
  year={2022}
}
```

## Dataset
You can download our dataset from Google Drive:

| Format                                                     | TO-Vanilla                                                   | TO-Crowd                                                     | TO-Scannet                                                   |
| ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ply** (point cloud)                                      | [Download (4.3GB)](https://drive.google.com/file/d/1oQ75oC-Q9BpmVQsON-qEvoe-oJbQkSsI/view?usp=sharing) | [Download (2.1GB)](https://drive.google.com/file/d/1s0WsuhNJABC2nE9hcZN3_rtSJoj2612t/view?usp=sharing) | [Download (4.3GB)](https://drive.google.com/file/d/1j4FWu09lTrF3euUA9PJ6fMhSclLlxyVh/view?usp=sharing) |
| **npz** (xyz, color, semantic_label, instance_label, bbox) | [Download (6.2GB)](https://drive.google.com/file/d/1ruXvISeADT3ERwcxjXgKfAINVy6KZJyy/view?usp=sharing) | [Download (2.8GB)](https://drive.google.com/file/d/1dgMjMUwT5DxhBDgIZYLqrP_OD_R2xf_V/view?usp=sharing) | [Download (6.5GB)](https://drive.google.com/file/d/12IVVEt5kUQrz0_Qis58TH6Fj4yr6cJTo/view?usp=sharing) |

Alternatively, for mainland China users, we also provide Baiduyun link:

| Format                                                     | TO-Vanilla                                                   | TO-Crowd                                                     | TO-Scannet                                                   |
| ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ply** (point cloud)                                      | [Download (4.3GB)](https://pan.baidu.com/s/1r0D6HnjHJC2eR-Ifwo6Iig?pwd=0000) | [Download (2.1GB)](https://pan.baidu.com/s/1jlrhJkFr00AXce55miRRmw?pwd=0000) | [Download (4.3GB)](https://pan.baidu.com/s/1wbynKbxWr4nNZLrBPSIAJw?pwd=0000) |
| **npz** (xyz, color, semantic_label, instance_label, bbox) | [Download (6.2GB)](https://pan.baidu.com/s/1yXlFv5X8byEkuoFtuSAVUg?pwd=0000) | [Download (2.8GB)](https://pan.baidu.com/s/1mMJhOo-6uXMR9RPz3snG0g?pwd=0000) | [Download (6.5GB)](https://pan.baidu.com/s/1D-_M6L4iu5S7eRGgXQI0RA?pwd=0000) |

## Code

We have provided an initial version of our code implementations at [here](./initial_code), including [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer) for the semantic segmentation task.

__Note__:
More details and other variants of our dataset, as well as more implementations of models, data pre-processing details are being organized with the best effort, and will be released soon, via a complete website!