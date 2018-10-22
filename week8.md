# 第八周作业：实现一个densenet的网络，并插入到slim框架中进行训练
**学员：** 冯学智

**手机：** 18210239929

**微信群昵称：** 北京-开发-冯学智

## 1.任务说明
**数据集:**

本数据集拥有200个分类，每个分类300张图片，共计6W张图片，其中5W张作为训练集，1W张图片作为验证集。图片已经预打包为tfrecord格式并上传到tinymind上。地址如下：
https://www.tinymind.com/ai100/datasets/quiz-w7

**模型:**

模型代码来自：
https://github.com/tensorflow/models/tree/master/research/slim

为了适应本作业提供的数据集，稍作修改，添加了一个quiz数据集以及一个训练并验证的脚本，实际使用的代码为：
https://gitee.com/ai100/quiz-w7-2-densenet

其中nets目录下的densenet.py中已经定义了densenet网络的入口函数等，相应的辅助代码也都已经完成，学员只需要check或者fork这里的代码，添加自己的densenet实现并在tinymind上建立相应的模型即可。

densenet论文参考 https://arxiv.org/abs/1608.06993
鼓励参与课程的学员尝试不同的参数组合以体验不同的参数对训练准确率和收敛速度的影响。

## 2.DenseNet模型
### 2.1对densenet的理解
**A.设计思路**
1. 核心思想是在保证网络中层与层之间最大程度的信息传输的前提下，直接将所有层连接起来。即靠后的处理层尽量多的保留原始特征信息，这一点是通过将前面多层的output同时作为后面层input来实现的（将不相邻的层连接起来）；**【稠密连接】**
2. 如果要实现前面每层的output同时作为某层的input，必须保证这些output的height和width的相同。显然不可能网络中所有层的heighthe和width都相同，所以作者提出了densenet中的block概念，每个block内部每层保持高宽一致，不同的block的高宽可以不同。**【块内稠密、shape相同，块间不同】**
3. block中每个layer不像传统卷积那样只有一个卷积层，而是每个layer都包括batch_normlization、activation function、Convalution or Pooling，训练时添加dropout操作，这些操作共同组成了一个layer，被称作BN-Relu-Conv。**【composite function】**
4. block间使用transition layer进行连接，每个transition layer包括convolution和pooling操作。**【transition layer】**
5. 如果块内每层的kernel数量为k，则输出的channel为k，又因为第l层的input是前面所有层的output，所以input的channel为k0 + k ×(l −1) ，显然k取一个较小值(网络较窄，参数数量就会少)，就可以达到其他网络的深层效果。这就是densenet的 **magic**,其中k被成为growth rate。**【growth rate】**
6. 即使块内每层只产生k个特征图，但靠后的层接收的input的数量依然非常大。为了减少运算量，在块内每层增加了1x1的卷积层(bottleneck layer)，不改变高宽，只缩减channels。即块内每层变为BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3).**【Bottleneck layer、DenseNet-B】**
7. 为了进一步提高模型紧凑度和计算效率，在每个transition layer只保留前一层的一部分，保留比例0<θ<=1。**【Compression、DenseNet-C、DenseNet-BC】**

**B.综合来讲，DenseNet有如下几个优点：**
1. 减轻了vanishing-gradient（梯度消失）
2. 加强了feature的传递
3. 更有效地利用了feature
4. 一定程度上较少了参数数量

### 2.2模型训练情况
具体训练模型：https://www.tinymind.com/SDMrFeng252/quiz-w8-densenet-2

**执行情况：**

在本地没有GPU的机器上做了调试，没有语法错误后提交至tinymind上执行。
1. 第一次执行，未将数据集中validation数据选中，而是选择了train数据，执行失败；
2. 第二次执行，载入点错选了train_image_classifier.py，应该是train_eval_image_classifier.py，主动中止；
3. 第三次执行，发现效果非常差，原因是每个block处理之后忘记赋值给net，导致网络基本没有得到训练，主动中止，修改后又提交；
4. 第四次执行，正常，但受限于tinymind每两小时停一次，两小时时被动停掉了；
5. 第五次执行，同上；
6. 发现第五次执行的后期，total_loss下降非常缓慢了，将learning_rate由0.1调整为0.01，继续训练看看，受限于作业提交时间，模型训练运行的log截图使用第五次执行最后的验证情况：
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/densenet-run5.jpg"/></div>

## 3.学习心得
**看第一遍录播视频，了解了本周内容的大体情况**

**A.** 一般的模型训练都分为三个阶段：
1. 特征抽提阶段
2. 特征整理层
3. 分类器

当然一般在进入特征抽提阶段前，都会对原始input做一些预处理。

**B.** 了解了slim工具包中各文件夹和各文件的组织结构，以及分别的用处

**C.** 小知识点
1. 各函数的组织及相互调用，如inception_v3_base的作用等；
2. 对shape不是299x299的图片的处理手段，及常规的预处理、数据增广的手段：裁切、左右翻转、饱和度、色相、对比度、亮度等随机处理，中央裁切的根据；
3. 模型前部的卷积为什么可以使用stride=2；
4. 连续两个3x3kernel VS. 一个5x5的kernel，技巧的原理；
5. 实现中使用了一处3x3的池化，比较特别，使用的理论根据；
6. 1x1kernel的卷积<==>全连接；
7. 连续1x3和3x1的卷积，替代3x3的卷积，用在feature size在[12~20]的中间层效果较好；
8. 结合网上的博客，了解了inception_v3发展的由来，和v2的区别之处等，使用到了对各branch输出的depth进行concate等；
9. 学习了利用tensorboard对训练情况进行分析，例如通过看total_loss变化，调整learning_rate的策略、sparcity的分析等;
10. 滑动平均的作用和意义，以及与正则化的区别；
11. checkpoint文件组成、分别存储的内容及相互关系；
12. slim框架下如何恢复模型，如何只利用训练好的模型中的部分结构；
13. 代码中scope的意义及作用；
14. 利用训练好的模型对图片进行预测的几种实现形式、freeze_graph等；
15. densenet的主要设计思想，及与其他模型的主要改进；

**看第二遍录播视频收获**
1. 上手了pycharm、tinymind、github等；
2. 将视频中的讲解到的案例在本机进行了练习，尤其是使用训练好的模型对图片进行预测的两种方法；
3. 结合论文、课程视频，深入理解了densenet的设计思路，复现了densenet。

**周六直播课的收获**
1. 使用特征重建数据；
2. style transfer的应用；
3. 对抗样本，即其影响：如交通安全方面坏影响；但可以利用对抗样本保护个人信息被自动采集；
4. nightmare风格应用；

**其他**
1. **学习ml、dl以来的感觉：模型参数过少，效果肯定不佳；模型参数过多，又会容易过拟合；模型层数太深，参数就多，容易引起梯度消失。所以机器学习是折中的艺术。**
2. 虽然课程给出了作业框架，但python基本功差，linux使用不熟练，浪费了不少时间，需要加强动手学习。
3. 机器学习、深度学习训练的是模型，debug过程训练的是人，让我们不再犯同样的错误。
