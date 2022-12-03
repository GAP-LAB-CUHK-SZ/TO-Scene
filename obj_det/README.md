## Object detection using adaptive sampling
### Environment
```
CUDA 11.1
pytorch 1.9.0
```
The following is required to be installed before running:
```
torvision 0.10.0
pyyaml
scipy
matplotlib
opencv-python
tensorboardX
```
Run the following commands to install pointnet2 and weighted FPS:
```angular2html
cd external
cd pointnet2
python setup.py install
cd ..
cd weighted_FPS
python setup.py install
```
### Data preparation
Download TOS dataset with bbox, instance label and semantic label. Then, modify the dataset_dir and save_dir under ./utils/generate_gt_heatmap.py.
Then run this script to generate the ground truth objectness map that is mentioned in the paper.
<br>
Then, put the TO-xxx-wHM folder under ./data
### Training
#### training on TO-crowd
For training the TO-crowd or TO-vanilla, it requires pretraining the heatmap module firstly,
to train the heatmap module, run the following commands:
```angular2html
python main.py --mode train --config ./configs/train_heatmap.yaml
```
Then, modify the hm_pretrain_path in ./configs/train_vote_adaptive_desk.yaml to point to the pretrain heatmap module's weight.
<br>
Finally run the adaptive votenet by:
```angular2html
python main.py --mode train --config ./configs/train_vote_adaptive_desk.yaml
```

## Original VoteNet baseline

If you wish to run original [VoteNet](https://github.com/facebookresearch/votenet) on our dataset, you may refer to the usage on [their official repo](https://github.com/facebookresearch/votenet) and our implementation [here](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene/tree/main/obj_det/votenet).