# 第十周作业：以VGG16为基础，构建一个FCN训练模型
**学员：** 冯学智

**手机：** 18210239929

**微信群昵称：** 北京-开发-冯学智

## 1.任务说明
**数据集:**

本作业使用Pascal2 VOC2012的数据中，语义分割部分的数据作为作业的数据集。

VOC网址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

其中本次作业使用VOC2012目录下的内容。作业数据集划分位于**VOC2012/ImageSets/Segmentation**中，分为train.txt 1464张图片和val.txt1449张图片。

语义分割标签位于**VOC2012/SegmentationClass**,注意不是数据集中所有的图片都有语义分类的标签。
语义分割标签用颜色来标志不同的物体，该数据集中共有20种不同的物体分类，以1～20的数字编号，加上编号为0的背景分类，该数据集中共有21种分类。编号与颜色的对应关系如下：
```py
# class
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
```


**模型代码:**

预训练模型使用tensorflow，modelzoo中的VGG16模型.

下载地址：http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

模型代码以课程视频week10 FCN部分的代码进行了修改，主要是代码整理，添加了数据输入和结果输出的部分。

代码参考：https://gitee.com/ai100/quiz-w9-code.git


模型参数的解释：

- checkpoint_path VGG16的预训练模型的目录，这个请根据自己建立的数据集的目录进行设置。
- output_dir 输出目录，这里使用tinymind上的/output目录即可。
- dataset_train train数据集的目录，这个请根据自己建立的数据集的目录进行设置。
- dataset_val val数据集的目录，这个请根据自己建立的数据集的目录进行设置。
- batch_size BATCH_SIZE，这里使用的是16,建立8X的FCN的时候，可能会OutOfMem，将batch_size调低即可解决。
- max_steps MAX_STEPS， 这里运行1500步，如果batch_size调整了的话，可以考虑调整一下这里。
- learning_rate 学习率，这里固定为1e-4, 不推荐做调整。

运行过程中，模型每100个step会在/output/train下生成一个checkpoint，每200步会在/output/eval下生成四张验证图片。

>FCN论文参考 https://arxiv.org/abs/1411.4038

## 2.作业过程
### 2.1数据准备
1. 下载了Pascal2 VOC2012数据集，并对数据集整体进行了了解，重点关注了SegmentationObject和SegmentationClass文件夹中的文件；

2. 阅读理解作业基础代码中的convert_fcn_dataset.py的内容，主要分为几步：
- 设置输出文件路径、源数据路径
- 拼接源数据文件名
- 以每个原图和label为一组，对每组进行编码，并输出到目标文件中（其中分类与颜色编码的映射关系较为关键）

3. 填充缺失代码两处：1.feature_dict参数补充；2.create_tf_record函数实现。
代码查看链接和截图如下：
https://github.com/SDMrFeng/quiz-w10-fcn/blob/master/convert_fcn_dataset.py
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/CSDN-quiz-week10-dataset-convert.png"/></div>

4. 执行上述修改的代码，生成fcn_train.record(403.48 MB)和fcn_val_record(400.06 MB)

5. 下载VGG16的预训练模型vgg_16.ckpt，并将以上三个数据集上传到tinymind数据集quiz-week10-dataset中，查看链接和截图如下：

https://www.tinymind.com/SDMrFeng252/datasets/quiz-week10-dataset
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/CSDN-quiz-week10-dataset-tinymind.png"/></div>

**注：** 在数据准备过程中主要是遇到了库依赖找不到的问题：使用pip安装了pydensecrf、opencv-python之后，因为本机存在多个python环境和conda环境，发现了pip和pip3安装位置不同，经过查资料解决了此问题。

### 2.2运行训练
1. 在复习录播视频的基础上，阅读了论文，论文中依次取pool5、pool4、pool3的feature map进行upsampling和拼接操作；
2. 在16X的FCN基础上，修改train.py文件，实现8X的FCN，代码查看链接和关键代码如下：
https://github.com/SDMrFeng/quiz-w10-fcn/blob/master/train.py
```Python
# 获取vgg16中pool4输出的feature map（在vgg16中pool4输出后将原图缩小为原来的1/16）
pool4_feature = end_points['vgg_16/pool4']
# 对pool4的输出进行1x1xnumber_of_classes(1x1x21)的卷积，不进行激活，并使用0初始化该层卷积核
with tf.variable_scope('vgg_16/fc8'):
    pool4_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                               activation_fn=None,
                               weights_initializer=tf.zeros_initializer,
                               scope='conv_pool4')

# Perform the upsampling(X2)
# 将最终的loigts大小由原图1/32变为1/16，以便与pool4的logits合并
# 反卷积长宽步长为2,即长宽会扩大2倍（倍数由upsample_factor指定）
upsample_factor = 2
upsample_filter_np_x2 = bilinear_upsample_weights(upsample_factor,
                                                  number_of_classes)
upsample_filter_tensor_x2 = tf.Variable(upsample_filter_np_x2,
                                        name='vgg_16/fc8/t_conv_x2')
upsampled_logits_16s = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2,
                            output_shape=tf.shape(pool4_logits_16s),
                            strides=[1, upsample_factor, upsample_factor, 1],
                            padding='SAME')

# Combine pool4_logits_16s with upsampled_logits_16s
upsampled_logits_16s = upsampled_logits_16s + pool4_logits_16s


# 获取vgg16中pool3输出的feauture map(在vgg16中pool3输出后将原图缩小为原来的1/8)
pool3_feature = end_points['vgg_16/pool3']
# 对pool3的输出进行1x1xnumber_of_classses(1x1x21)的卷积，不进行激活，并使用0初始化该核
with tf.variable_scope('vgg_16/fc8'):
    pool3_logits_8s = slim.conv2d(pool3_feature, number_of_classes, [1, 1],
                               activation_fn=None,
                               weights_initializer=tf.zeros_initializer,
                               scope='conv_pool3')

# Perform the upsampling(X2)
# 将上采样后的upsampled_logits_16s由原图1/16变为1/8，以便与pool3的logits合并
# 反卷积长宽步长为2,即长宽会扩大2倍（倍数由upsample_factor指定）
upsample_factor = 2
upsample_filter_np_x2 = bilinear_upsample_weights(upsample_factor,
                                                  number_of_classes)
upsample_filter_tensor_x2 = tf.Variable(upsample_filter_np_x2,
                                        name='vgg_16/fc8/t_conv_x2_x2')
upsampled_logits_8s = tf.nn.conv2d_transpose(upsampled_logits_16s,
                               upsample_filter_tensor_x2,
                               output_shape=tf.shape(pool3_logits_8s),
                               strides=[1, upsample_factor, upsample_factor, 1],
                               padding='SAME')

# Combine pool3_logits with upsampled_logits
upsampled_logits = upsampled_logits_8s + pool3_logits_8s


# Perform the upsampling(X8)
# 将上采样后的upsampled_logits由原图1/8变为与原图等大
# 反卷积长宽步长为8,即长宽会扩大8倍（倍数由upsample_factor指定）
upsample_factor = 8
upsample_filter_np_x8 = bilinear_upsample_weights(upsample_factor,
                                                  number_of_classes)
upsample_filter_tensor_x8 = tf.Variable(upsample_filter_np_x8,
                                        name='vgg_16/fc8/t_conv_x8')
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits,
                              upsample_filter_tensor_x8,
                              output_shape=upsampled_logits_shape,
                              strides=[1, upsample_factor, upsample_factor, 1],
                              padding='SAME')
```
3. 在本地运行，确保无语法错误，且能正确训练几十个step之后，将相关代码上传至github，并在tinymind上新建模型quiz-w10-fcn。模型查看链接：
https://www.tinymind.com/SDMrFeng252/quiz-w10-fcn

4. 在tinymind上新建运行对模型进行训练，第一次执行就顺利完成，但结束后发现output中没有eval输出，在train中也没有最后5个整百的checkpoint文件。对比一下本地和tinymind运行的输出，发现情况如下：
<div align=center><img src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/CSDN-quiz-week10-output-exception.png.jpg"/></div>

5. 分析4.中出现的问题：第一次进入if gs%10==0前gs没有取得应有的值1，运行if block第一个语句时，每次gs都增加1，导致进入不到100和200两个if block中，所有没有得到应有的ckpt文件和eval输出。理论上gs值在if block内不会变化，因为gs值在block内增加了1，导致出现了上图的情形。

6. 解决4.中问题：十分曲折，上网查询、仔细分析代码、微信群咨询、自己摸索，调整n次代码、提交版本库、重新运行n次。测试了十几次，每次运行启动时间较长，总共发了大半天的时间，最后发现是tensorflow版本的问题，最开始使用的是tf 1.8,而1.4运行直接报错，1.7和1.8会出现同样现象。最后使用tf 1.6运行成功，运行链接：

https://www.tinymind.com/executions/vgs77g5t

以下给出tinymind运行的的log截图：
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_quiz_week10_run_output_log.png"/></div>

### 2.3执行的输出
执行的输出连接：https://www.tinymind.com/executions/vgs77g5t/output

以下给出tinymind运行的的output截图：
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_quiz_week10_run_output_files.png"/></div>

以下给出tinymind运行一个预测的拼图：
|   |第1列 |第2列|
|---|---|---|
| 第1行  | 原图  | annotation  |
| 第2行  | prediction  |  prediction_crfed |
| 第3行  |   | overlay  |
<div align=center><img src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_quiz_week10_run_output_example.png"></div>


## 3.学习心得
1. 本周课程录播内容每一节都是快听一遍，细听一遍并仔细做笔记，理解比较透彻，作业实现时也比较顺利。
2. 本周课程里介绍到的关于FCN的几个重要思想。
- 全连接要求输入的spatial必须是固定的，用卷积代替全连接之后，对输入的尺寸就没有了限制；
- 使输出的feature map更稠密(一维向量变成热力图)的措施：扩大输入的尺寸、将padding置为same。这样操作之后，可以提高Semantic Segmentation的准确度。
- 对feature map进行双线性插值后，再进行反卷积操作(upsampling)
- 对输入的图片进行padding预处理，size处理成32的倍数（以免卷积过程中取整截断，反卷积之后不能恢复原图的size）
- 在VGG中，减少pooling层，也可以达到提高输出feature map稠密度的效果，但副作用是feature map中的每个元素在原图上对应的感受野会下降，这是不希望看到的；同时预训练模型中的权重也不适用了，同时参数量增加、计算量增加；
- 在VGG中，使用跳接结构，充分利用中间层输出的feature map：对靠后层的feature map进行upsample操作，再与临近上层的feature map合并，逐步向前进行upsample操作，这样避免了直接从最后的特征图直接放大导致的图像失真，提高了准确度。(结构清晰明了，利于理解)
3. 多孔卷积的重点：
- a.将pooling层的stride置为1，feature map的size变为stride=2时的2倍，但特征图上的每个元素在原图上的感受野变为原来的1/2；
- b.将pooling后的第一个conv层使用多空卷积来代替(rate=2,padding='SAME'),相当于将特征图在原图上的感受野变为原来的2倍；
- 综合a和b，特征图中元素在原图中的感受野大小没有变化，但特征图的稠密度变为原来的2倍，而且原预训练网络中的大部分权重都还能使用；
4. Deeplab的ASPP思想：在不同感受野范围内进行信息捕捉，实现多种空间尺度上的信息综合，提高判断位置和分类的准确性。
5. GCN：采用大卷积核对几乎整张图进行卷积，可以更准确得到对整张图的描述（借鉴了多种结构，计算量比较大），其中GCN块(Global Convolution Network)和BR块(Boundary refinement)是整个模型的重要组成结构。


PS:在tinymind上执行训练调问题时浪费了时间，归根结底是不够游刃有余、头脑不够清晰，需要大胆细心。
