# Semantic-Aware Auto-Encoders for Self-Supervised Representation Learning


This is the code of our paper "Semantic-Aware Auto-Encoders for Self-Supervised Representation Learning".


[Guangrun Wang](https://wanggrun.github.io), [Yansong Tang](https://andytang15.github.io/), [Liang Lin](http://www.linliang.net/), and [Philip H.S. Torr](https://www.robots.ox.ac.uk/~phst/).


## Project Page:

[Project Page](https://wanggrun.github.io/projects/works/saae)


A highly recommended project: [Traditional Classification Neural Networks are Good Generators](https://wanggrun.github.io/projects/works/cag)


## ImageNet


An example of SSL training script on ImageNet. More and larger GPUs are better. Please refer to [DINO](https://github.com/facebookresearch/dino) to use multi-node training).



```shell
python -m torch.distributed.launch --nproc_per_node=8 main_saae.py --arch vit_base --data_path xxxxxxxxx/ILSVRC2012/train --output_dir  xxxxxxx/name_of_output_dir/   --epochs  400   --reference_crops_number  1  --batch_size_per_gpu 120  --use_fp16 true  --num_workers  4   --weight_decay  0.1    --weight_decay_end   0.1  --use_bn_in_head true   --lr  2.4e-4

```


An example of linear evaluation script on ImageNet:


```shell
python3  -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --arch vit_base --checkpoint_key student --num_workers 4 --data_path  xxxxxxxx/ILSVRC2012    --pretrained_weights   xxxxxxxxx/name_of_output_dir/checkpoint.pth  --output_dir   xxxxxxxxxxxx/name_of_eval_output_dir/  --lr 0.01

```


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


# Installation


This repo has been tested in the following environment. More precisely, this repo is a modification on the DINO. Installation and preparation follow that repo. Please acknowledge the great work of the team of DINO. Thanks for their outstanding contributions.


Pytorch1.9


[DINO](https://github.com/facebookresearch/dino)

# Related Project


[Solving Inefficiency of Self-supervised Representation Learning](https://github.com/wanggrun/triplet)

