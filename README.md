# mobilenetv2_deeplabv3_pytorch
From [tensorflow/models/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab), we can know details of Deeplab v3+ ([paper](https://arxiv.org/abs/1802.02611)).


The [TensorFlow DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) provides four pre_train models. Using **Mibilenetv2** as feature exstractor and according to [offical demo](https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb) (run on [Calab](https://colab.research.google.com/notebooks/welcome.ipynb)), I have given a tensorflow segmentation demo in my [demo_mobilenetv2_deeplabv3](https://github.com/lizhengwei1992/demo_mobilenetv2_deeplabv3).


These codes are implementation of mobiletv2_deeplab_v3 on pytorch.

## network architecture
In [demo_mobilenetv2_deeplabv3](https://github.com/lizhengwei1992/demo_mobilenetv2_deeplabv3), use function ```save_graph()```
to get tensorflow graph to folder pre_train, then run ```tensorboard --logdir=pre_train``` to open tensorboard in browser:
![tensorboard](https://github.com/lizhengwei1992/mobilenetv2_deeplabv3_pytorch/raw/master/images/tensorboard.png)

the net architecture mainly contains: **mobilenetv2**、**aspp**.


<div align=center>
      
![graph](https://github.com/lizhengwei1992/mobilenetv2_deeplabv3_pytorch/raw/master/images/graph.png)
      
</div>
      
      
      
      
### mobilenetv2
the mobilenetv2 in deeplabv3 is little different from original architecture at output stride and 1th block.
Attention these blockks (1th 4th 6th) in [code](https://github.com/lizhengwei1992/mobilenetv2_deeplabv3_pytorch/blob/master/model/MobileNet_v2.py) .

    +-------------------------------------------+-------------------------+
    |                                               output stride
    +===========================================+=========================+
    |       original MobileNet_v2_OS_32         |          32             | 
    +-------------------------------------------+-------------------------+
    |   self.interverted_residual_setting = [   |                         |
    |       # t, c, n, s                        |                         |
    |       [1, 16, 1, 1],                      |  pw -> dw -> pw-linear  |
    |       [6, 24, 2, 2],                      |                         |
    |       [6, 32, 3, 2],                      |                         |
    |       [6, 64, 4, 2],                      |       stride = 2        |
    |       [6, 96, 3, 1],                      |                         |
    |       [6, 160, 3, 2],                     |       stride = 2        |
    |       [6, 320, 1, 1],                     |                         |
    |   ]                                       |                         |
    +-------------------------------------------+-------------------------+
    |          MobileNet_v2_OS_8                |          8              |
    +-------------------------------------------+-------------------------+
    |   self.interverted_residual_setting = [   |                         |
    |       # t, c, n, s                        |                         |
    |       [1, 16, 1, 1],                      |    dw -> pw-linear      |
    |       [6, 24, 2, 2],                      |                         |
    |       [6, 32, 3, 2],                      |                         |
    |       [6, 64, 4, 1],                      |       stride = 1        |
    |       [6, 96, 3, 1],                      |                         |
    |       [6, 160, 3, 1],                     |       stride = 1        |
    |       [6, 320, 1, 1],                     |                         |
    |   ]                                       |                         |
    +-------------------------------------------+-------------------------+

# TODO

- add test codes
- add pre_train model





