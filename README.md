# DeepLearningKnowledge

> 深度学习相关算法总结

[TOC]



#1. 框架相关(PyTorch,Tensorflow,Jax等)

## 1.1 动态图和静态图的差别(模型框架的差别)

> https://www.zhihu.com/people/HiFuture_ToMyDream/posts/posts_by_votes

`PyTorch` 是一种动态构建计算图的深度学习框架。计算图是用来描述运算的有向无环图，节点表示数据，边表示操作或者运算。

![img](https://gitee.com/wanghui88888888/picture/raw/master/img/v2-b8609ea5925eb6c32e0279d90516ee9a_720w.jpg)

由用户手动创建的Tensor称为叶子节点，默认不计算梯度；由funcation函数计算得到的Tensor是非叶子节点。

在整个计算图中，只要有一个节点`requires_grad = True` ,所有依赖该节点的节点的`requires_grad`也为True。

PyTorch采用的是动态图机制(Dynamic Computational Graph)，Tensorflow1.x采用的是静态图机制(Static Computational Graph)。

**动态图是运行和搭建同时运行，优点是灵活，易调试，一边构图，一边运算数据。在Pytorch中叶子节点是用户创建的变量，在PyTorch的实现中，为了节省内存，在梯度反向传播后，非叶子节点的梯度都会被释放掉。**

**静态图是先搭建图，然后在输入数据进行运算。优点是高效，因为静态图是先定义后运行的方式，之后再运行的时候就不需要重新构建计算图，速度比动态图更快，但是非常不灵活，不便于进行调试。静态图例如Tensorflow1.x和JAX。所以tensorflow会给预先传入的数据设定为`tf.placeholder`作为构建图之后的输入数据的占位 **

> https://www.zhihu.com/question/330766768/answer/924812910

> https://www.jianshu.com/p/505e1e0142c1

![image-20220318115239927](https://gitee.com/wanghui88888888/picture/raw/master/img/image-20220318115239927.png)

---

![image-20220318120119996](https://gitee.com/wanghui88888888/picture/raw/master/img/image-20220318120119996.png)

---

 ## 1.2 PyTorch模型实现

`torch.nn.Module`是所有神经网络模块的基类，所有神经网络模型都应该继承这个基类，所有的模型都是`torch.nn.Module`的子类，并进行重载`init`(初始化)和`forwaed`(前向传播)函数。每个类都有一个对应的`nn.funcational`函数，类定义了所需要的`arguments`和模块的`parameters` ,在forward函数中将`arguments`和`parameters`传给`nn.functional`的对应的函数来实现forward功能。

**流程：nn.Module 在被调用的时候，一般是以 module(input) 的形式，此时会首先调用** **`self.__call__`，接下来这些 hooks 在模块被调用时候的执行顺序如下图所示：**

![img](https://gitee.com/wanghui88888888/picture/raw/master/img/v2-bb042caa6b90b4b593ba849a25cfa70a_720w.jpg)



#2. 模型训练相关

## 2.1 模型正则化

`Layer Normalization` 和 `Batch Normalization`的区别

> https://zhuanlan.zhihu.com/p/74516930

![image-20220226191902683](https://gitee.com/wanghui88888888/picture/raw/master/img/image-20220226191902683.png)

![image-20220226192004880](https://gitee.com/wanghui88888888/picture/raw/master/img/image-20220226192004880.png)



## 2.2 模型训练loss变成NAN

**1. 梯度爆炸。解决方法：调学习率、梯度剪裁、归一化**

**2. 计算loss的时候有log0，可能是初始化的问题，也可能是数据的问题**

**模型训练过程中出现`NAN`的本质原因是是出现了`下溢出` 和 `上溢出` 的现象**

- `上溢出`首先怀疑模型中的指数运算， 因为模型中的数值过大，做exp(x)操作的时候出现了上溢出现象，这里的解决方法是推荐做`Nrom` 操作，对参数进行正则化，这样在做exp操作的时候就会很好的避免出现上溢出的现象，可以做`LayerNorm``BatchNorm` 等，这里我对模型加fine-tune的时候使用`LayerNorm` 解决了loss为`NAN` 的问题。【比如不做其他处理的softmax中分子分母需要计算exp（x），值过大，最后可能为INF/INF，得到NaN，此时你要确认你使用的softmax中在计算exp（x）做了相关处理（比如减去最大值等等)】

`上溢出` 同样可能是因为![[公式]](https://www.zhihu.com/equation?tex=++), x/0的原因，这样就不是参数的值过大的原因，而是具体操作的原因，例如，在自己定义的softmax类似的操作中出现问题，下面是softmax解决上溢出和下溢出的解决方法：

![img](https://gitee.com/wanghui88888888/picture/raw/master/img/v2-b31c6a82a3d2246c2c3b7e4fabfb53c4_720w.jpg)

- `下溢出` 下溢出一般是![[公式]](https://www.zhihu.com/equation?tex=\log(0)) 或者exp(x)操作出现的问题。可能的情况可能是学习率设定过大，需要降低学习率，可以降低到学习率直至不出现nan为止，例如将学习率1e-4设定为1e-5即可。

除了降低学习率的方法，也可在在优化器上面加上一个eps来防止分母上出现0的现象，例如在batchnorm中就设定eps的数值为1e-5，在优化器同样推荐加入参数`eps`,`torch.optim.adam`中默认的`eps` 是`1e-8`。但是这个值属实有点小了，可以调大这个默认的`eps` 值，例如设定为`1e-3`。

```python
optimizer1 = optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
optimizer1 = optim.Adam(model.parameters(), lr=0.001, eps=1e-3)
optimizer2 = optim.RMSprop(model.parameters(), lr=0.001, eps=1e-2)
```

训练过程中loss为`nan`的情况，也包括下面几种

- 输入中就含有NaN。数据集不完整，例如训练的data或者使用的label为空，就会导致这种情况，label缺失问题也会导致loss一直是nan，需要检查label。
- 梯度爆炸，注意每个batch前梯度要清零，optimizer.zero_grad()。
- 当我们使用具有 log() 的损失函数，如 Focal Loss 或 Cross Entropy 时，输入张量的某些维度可能是一个非常小的数字，正常情况是float32是一个极其小的数但也不是0，但是amp使用半精度可能就直接摄入到了0，因此就出现NAN的问题了。因此，我们可以在涉及到`log` 操作的时候，将float16转化为float32




# 3. Bert相关









