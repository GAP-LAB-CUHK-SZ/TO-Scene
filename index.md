## TO-Scene

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



### Data

Please visit our main project repository for more information and access to code, data: https://github.com/GAP-LAB-CUHK-SZ/TO-Scene

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h59lkm18dqj20u015ugx8.jpg" alt="image-20220817112242837" style="zoom:80%;" />

### Benchmarks

Coming soon...

### News

* 2022-08-17: Benchmark Challenge available

* 2022-08-17: TO-Scene initial release
