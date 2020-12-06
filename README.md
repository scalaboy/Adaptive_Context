# python train.py 训练即可。数据下载NWPU数据即可。
this code is the official code for the paper :Adaptive Context Learning Network for
Crowd Counting.which accepted by the IEEE SMC 2020.




Abstract—The task of crowd counting is to estimate the
accurate number of people in photos taken from unconstrained
surveillance scenes. It is in general a challenging problem due to
the input scale variations and perspective distortions. Previous
methods make efforts to enhance the representation ability by
using multi-scale features of the scene pictures. However, most
of these methods directly add or fuse the features, in which
the influences of different feature sizes are equally considered.
In this paper, we propose a novel architecture called adaptive
context learning network (ACLNet) to incorporate context of
features in multiple levels. In this architecture, the original
image features are enhanced by a multi-level feature generating
module, and then the multi-level features are up-sampled to the
same size and re-weighted for fusing. The ACLNet incorporates
the context information existed in sub-regions of various scales
adaptively, thus it is able to enhance the representative ability
of multi-level features. We perform several experiments on
public ShanghaiTech (A and B), UCF CC 50 and NWPU-crowd
datasets. Our proposed ACLNet achieves the state-of-the-art
results compared with existing methods.
Index Terms—crowd counting, density map, pyramid pooling,
adaptive convolution

详情博客见公众号：Agent的潜意识

Citation
If you find this project is useful for your research, please cite:


@article{gao2019c,
  title={Adaptive Context Learning Network for Crowd Counting},
  author={Zhao Liu 1 , Guanqi Zeng ∗ , Zunlei Feng 2 , Rong Zhang 1 , Mingli Song 2 , Jianping Shen},
  journal={IEEE SMC2020},
  year={2020}
}
