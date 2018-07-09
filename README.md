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
      

