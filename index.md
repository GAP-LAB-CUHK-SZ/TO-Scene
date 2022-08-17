# TO-Scene

Many basic indoor activities such as eating or writing are always conducted upon different tabletops (e.g., coffee tables, writing desks). It is indispensable to understanding tabletop scenes in 3D indoor scene parsing applications. Unfortunately, it is hard to meet this demand by directly deploying data-driven algorithms, since 3D tabletop scenes are rarely available in current datasets. 

To remedy this defect, we introduce TO-Scene, a large-scale dataset focusing on tabletop scenes, which contains 20,740 scenes with three variants. To acquire the data, we design an efficient and scalable framework, where a crowdsourcing UI is developed to transfer CAD objects from ModelNet and ShapeNet onto tables from ScanNet, then the output tabletop scenes are simulated into real scans and annotated automatically.

More information can be found in our [paper](https://arxiv.org/abs/2203.09440).

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h593b8omhtj22yn0u0du0.jpg" alt="pipeline_new.jpg"  />

Our work has been published to ECCV2022 (Oral). If you use the TO-Scene data or code please cite:

```
@inproceedings{xu2022toscene,
  title={TO-Scene: A Large-scale Dataset for Understanding 3D Tabletop Scenes},
  author={Xu, Mutian and Chen, Pei and Liu, Haolin and Han, Xiaoguang},
  booktitle={ECCV},
  year={2022}
}
```



# Data

Please visit our main project repository for more information and access to code, data: [Here](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene)

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h59lkm18dqj20u015ugx8.jpg" alt="image-20220817112242837" style="zoom:67%;" />



# Benchmarks

## Overview

We furthermore provide with the data also a benchmark suite covering different aspects of semantic scene understanding at different levels. To ensure unbiased evaluation of these tasks, we follow the common best practice to use a server-side evaluation of the test set results, which enables us to keep the test set labels private.

Test set evaluation is performed using [CodaLab competitions](https://competitions.codalab.org/competitions/). For each task, we setup a competition handling the submissions and scoring them using the non-public labels for the test set sequences. See the individual competition websites for further details on the participation process. Here, we will only provide the metrics and the leaderboards.

## 3D Semantic Segmentation Benchmark

See our [competition website](https://competitions.codalab.org/competitions/) for more information on the competition and submission process.

**Evaluation and metrics**

Our evaluation ranks all methods according to the PASCAL VOC intersection-over-union metric (IoU). IoU = TP/(TP+FP+FN), where TP, FP, and FN are the numbers of true positive, false positive, and false negative pixels, respectively. Predicted labels are evaluated per-vertex over the respective 3D scan mesh; for 3D approaches that operate on other representations like grids or points, the predicted labels should be mapped onto the mesh vertices.

**Leaderboard**

Coming soon...

## 3D Instance Detection Benchmark

See our [competition website](https://competitions.codalab.org/competitions/) for more information on the competition and submission process.

**Evaluation and metrics**

Our evaluation ranks all methods according to the average precision for each class. We report the mean average precision AP at overlap 0.25 (AP 25%), overlap 0.5 (AP 50%), and over overlaps in the range [0.5:0.95:0.05] (AP). Note that multiple predictions of the same ground truth instance are penalized as false positives.

**Leaderboard**

Coming soon...



# News

* 2022-08-17: Benchmark Challenge available

* 2022-08-17: TO-Scene initial release
