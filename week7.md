# 第七周作业：在mnist数据集上训练卷积神经网络
**学员：** 冯学智

**手机：** 18210239929

**微信群昵称：** 北京-开发-冯学智

**任务：**

1. 理解掌握卷积的前向传播过程，准确实现padding=same的填补，并计算input经filter之后output.

2. 使用tensorflow，构造并训练一个卷积神经网络，在mnist数据集上训练神经网络，在测试集上达到超过98%的准确率。

**作业要点提示：**

使用tensorflow，构造并训练一个神经网络，在测试机上达到超过98%的准确率。
在完成过程中，需要综合运用目前学到的基础知识：
- 深度神经网络
- 激活函数
- 正则化
- 初始化
- 卷积
- 池化

并探索如下超参数设置：
- 卷积kernel size
- 卷积kernel 数量
- 学习率
- 正则化因子
- 权重初始化分布参数

## 1.卷积的前向传播
**1.1 padding=same填补**

```
########################## My code here ##########################
in_height = in_s[1]
in_width = in_s[2]
filter_height = f_s[0]
filter_width = f_s[1]
#1.计算输出的高和宽
#out_height = ceil(float(in_height) / float(stride))
#out_width  = ceil(float(in_width) / float(stride))
#2.计算需要填补的高和宽
if (in_height % stride == 0):
    pad_along_height = max(filter_height - stride, 0)
else:
    pad_along_height = max(filter_height - (in_height % stride), 0)
if (in_width % stride == 0):
    pad_along_width = max(filter_width - stride, 0)
else:
    pad_along_width = max(filter_width - (in_width % stride), 0)
#3.计算上下左右各需填补的行列数
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left

in_batch = in_s[0]
in_channels = in_s[3]
#4.先初始化上下填补的四维数组，拼接到一起
temp_top = np.zeros((in_batch, pad_top, in_width, in_channels))
temp_bottom = np.zeros((in_batch, pad_bottom, in_width, in_channels))
temp = np.concatenate((temp_top, input, temp_bottom), axis = 1)
#5.初始化左右侧填补的四维数组，拼接到一起
temp_left = np.zeros((in_batch, in_height + pad_along_height,
                  pad_left, in_channels))
temp_right = np.zeros((in_batch, in_height + pad_along_height,
                  pad_right, in_channels))
temp = np.concatenate((temp_left, temp, temp_right), axis = 2)
##################################################################
input = temp
in_s = input.shape

```

**1.2 filter对input的处理**
```
############################# My code here ############################
out_batch = out_shape[0]
out_height = out_shape[1]
out_width = out_shape[2]
out_channels = out_shape[3]
for b in np.arange(out_batch):  # 批次不变,等于input的batch
   for i in np.arange(out_height):
       for j in np.arange(out_width):
           for k in np.arange(out_channels): # channels不变(=filter的channels)
               # 感受野的height方向位置范围是从i*stride到i*stride+f_height
               #        width方向位置范围是从j*stride到j*stride+f_width
               # input的每个channel都与卷积核进行卷积，然后求和得到一个值
               output[b, i, j, k] = np.sum(np.multiply(
                   input[b,
                         i * stride : i * stride + f_s[0],
                         j * stride : j * stride + f_s[1],
                         :],
                   filter[:, :, :, k]))
#######################################################################
```

## 2.在mnist数据集上训练神经网络（使用slim框架）
**2.1整体思路**

从课程对各种框架的应用的例子中，选中slim框架的例子作为本次作业的基础。对例子做以下调整，提高模型的预测准确率
- 每层卷积之后跟池化操作，调整卷积kernel的数量及size
- 激活函数使用relu
- 增加正则化，并调节正则化因子
- 手工调整权重初始化分布参数
- 调整学习率

**2.2参数调整过程**

1. 例子中在输出准确率和loss时，也对测试集使用了dropout。模型对测试集进行预测时，将keep_prob设置为1，更能准确的评估模型效果；

2. 增加step数量

例子中使用了3000个step（5 epochs）,根据以往经验，增加step，可以改善模型训练效果，因此先增加step看看效果：
此时，其他参数设置：两个卷积核（5*5*32、5*5*64），默认的权重初始化，学习率0.01，正则因子7e-5

| step数量|  准确率     |
| :--:|  :---------  |
| 3000     |   0.9557  |
| 9000     |   0.9781  |
| 21000    |   0.9858  |

接下来的所有训练中，均使用21000step（35 epochs）

3. 调整权重初始化分布参数

增加了slim的权重初始化分布设置，提升了模型的收敛速度，但试了几个不同的参数，并没有影响模型最后的预测准确率（因该是step数量比较大，保证了预测的稳定性）
```
h_conv2 = tf.contrib.slim.conv2d(h_pool1, 64, [5,5],
              padding='SAME',
              activation_fn=tf.nn.relu,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
```

4. 调整学习率

此时，其他参数设置：两个卷积核（5*5*32、5*5*64），权重初始化使用的均值0、方差0.01的正太分布，正则因子7e-5，21000个step

| 学习率|  准确率     |
| :--:|  :---------  |
| 0.01   |   0.9838  |
| 0.02   |   0.9883  |
| 0.03   |   0.9909  |
| 0.1    |   0.9926  |
| 0.3    |   0.9916  |

接下来的所有训练中，均使用0.1作为学习率

5. 正则因子调整

此时，其他参数设置：两个卷积核（5*5*32、5*5*64），权重初始化使用的均值0、方差0.01的正太分布，学习率0.1，21000个step

| 正则因子|  准确率     |
| :--:|  :---------  |
| 7e-5   |   0.9926  |
| 1e-5   |   0.9913  |
| 1e-6   |   0.9931  |
| 无正则  |   0.9925  |

接下来的所有训练中，均使用1e-6作为正则因子

6. 卷积kernel size和数量调整

此时，其他参数设置：权重初始化使用的均值0、方差0.01的正太分布，学习率0.1，正则因子1e-6，21000个step

| kenerl数量 | kenerl Size |  准确率     |
| :--:|  :---------  |  :---------  |
| 3  |5×5×32 5×5×64 5×5×128 |   0.9926  |
| 2  |5×5×64 5×5×128|   0.993  |
| 2  |3×3×32 3×3×64|   0.9843  |
| 2  |5×5×32 5×5×64|   0.9931  |
| 2  |7×7×32 7×7×64|   0.9935  |
| 2  |7×7×32 5×5×64|   0.9943  |

综上来看，在使用两个kenerl，size分别为7×7×32 5×5×64时，效果较好。

具体训练过程：https://www.tinymind.com/SDMrFeng252/mnist-multi-layer-nn-slim/executions

## 3.心得体会
1. 通过实现same padding的逻辑，加深了对padding的认识；
2. 通过实现对input实施卷积kernel运算，更深入的理解的卷积运算的过程，及input到output的shape变化；
3. 实现卷积神经网络可使用不同的写法(框架)，但逻辑思想都是一致的，不同之处在于参数的使用方式；因此重要的是学会怎么样根据模型的训练结果，通过调整参数来优化模型；
4. 本周作业使用卷积使预测准确率达到了0.993，甚至更高（输入是28*28）；上周使用全连接神经网络，预测准确率达到0.98以上已经不容易（输入是1*784）；感觉上周的输入格式丢失了列于列之间的相关性，本周使用卷积保留了列于列之间的相关性，在增加了计算量的同时，提高了预测的准确率。
