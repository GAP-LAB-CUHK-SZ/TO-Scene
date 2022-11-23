# 3D Semantic Segmentation.
This repository runs [Point Transformer](https://arxiv.org/abs/2012.09164) on TO-Scene for 3D semantic segmentation task.
The codebase comes from a reproduction at [here](https://github.com/POSTECH-CVLab/point-transformer).

---
## Environment

1. Dependencies:
- Hardware: 4GPUs (RTX 3090)
- Software: PyTorch>=1.5.0, Python>=3, CUDA>=11.0, tensorboardX, sharedarray, h5py, pyYaml.

2. Compile pointops:

Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (Note that if you install cuda by conda, it won't provide nvcc and you should install cuda manually). Then, compile and install pointops as follows:
```
cd lib/pointops
python3 setup.py install
```

---
## Dataset preparation
- Download our dataset following the [README](../README.md) at the main page:
---

## Usage
- Train

  - Specify the data_root and split_root location in config, and remember to modify the dataset variants of tovanilla/tocrowd/toscannet, then do training as follows:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tool/train.sh toscene \
    [tovanilla/tocrowd/toscannet]_pointtransformer
    ```

- Test

  - Afer training, you can test the checkpoint as follows:

    ```
    CUDA_VISIBLE_DEVICES=0 sh tool/test.sh toscene \ 
    [tovanilla/tocrowd/toscannet]_pointtransformer
    ```

- Run your own model
  - Youo can also run your own model on TO-Scene, by simply employing the data extraction in [tovanilla_crowd.py](./util/tovanilla_crowd.py) for running on TO_Vanilla or TO_Crowd and [toscannet.py](./util/toscannet.py) for running on TO_ScanNet.

- Result visualization
  - You may refer to [vis_tovanilla.py](./vis_vanilla.py) and [vis_tocrowd.py](./vis_tocrowd.py) for visualizing the final evaluation results.

---
## Acknowledgement
The code is based on [Point Transformer](https://arxiv.org/abs/2012.09164) at [here](https://github.com/POSTECH-CVLab/point-transformer).
We also refer [PAConv repository](https://github.com/CVMI-Lab/PAConv).
