# 第九周作业：利用slim框架和object_detection框架，做一个物体检测的模型
**学员：** 冯学智

**手机：** 18210239929

**微信群昵称：** 北京-开发-冯学智

## 1.任务说明
**数据集:**

课程准备好的一个数据集，拥有5个分类，共155张图片，每张图片都做了标注，标注数据格式与voc数据集相同。数据地址如下：
https://gitee.com/ai100/quiz-w8-data.git

数据集中的物品分类如下：
- computer
- monitor
- scuttlebutt
- water dispenser
- drawer chest

数据集中各目录如下
- images， 图片目录，数据集中所有图片都在这个目录下面。
- annotations/xmls, 针对图片中的物体，使用LabelImg工具标注产生的xml文件都在这里，每个xml文件对应一个图片文件，每个xml文件里面包含图片中多个物体的位置和种类信息。

**模型代码:**

本次代码使用tensorflow官方代码，代码地址如下：
https://github.com/tensorflow/models/tree/r1.5
> 因为最新的代码做了一些变化，需要使用pycocotool这个库，但是这个库的安装很复杂，目前暂时无法在tinymind上进行，所以这里使用比较老版本的代码

主要使用的是research/object_detection目录下的物体检测框架的代码。

（预训练模型）object_detection框架提供了一些预训练的模型以加快模型训练的速度，不同的模型及检测框架的预训练模型不同，常用的模型有resnet，mobilenet以及最近google发布的nasnet，检测框架有faster_rcnn，ssd等，本次作业使用mobilenet模型ssd检测框架，其预训练模型请自行在model_zoo中查找:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

>ssd论文：https://arxiv.org/abs/1512.02325

>mobilenet论文：https://arxiv.org/abs/1704.04861

## 2.作业过程（主要按照作业要点提示）
### 2.1数据准备
1. 在本机安装了labelImg工具，并做了一些练习，了解了labelImg的整体功能和快捷键，安装过程中遇到一些坑，主要和本机环境配置有关，都666自行解决；
2. 复制一份create_pet_tf_record.py文件，生成一个create_quiz_w8_tf_record.py文件，并删除文件中与mask、faces_only相关的代码，删除get_class_name_from_filename函数；
3. 阅读代码，发现数据集少一个文件trainval.txt，分析之后发现此文件存储了所有图片文件名称(不包含后缀)，写了一小段代码，生成了此文件；
4. **运行生成tf_record文件**，得到pet_train.record和pet_val.record两个文件，上一步中未修改训练集和验证集样本数量比例，因此测试集中有108个样本，验证集中有47个样本；在此过程中遇到一些问题，通过666和再读几遍作业指南，都自行化解。
5. 从上述model_zoo链接中下载了名为“ssd_mobilenet_v1_coco”的预训练模型（mobilenet模型ssd检测框架的预训练模型有几个，不清楚哪个最好，随便选了一个）
6. 将准备好的数据集上传到tinymind中：https://www.tinymind.com/SDMrFeng252/datasets/quiz-week9-dataset

注：在此数据集之前，上传了一个名为quiz-w9-dataset的数据集，但由于其中ssd_mobilenet_v1_pets.config中参数设置有误，重建同名数据时，模型执行异常，故新建quiz-week9-dataset数据集。

截图如下：
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_week9_dataset.png"/></div>

### 2.2运行训练
1. 对文件models/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config进行修改，将num_classes改为5，num_examples改为47，**PATH_TO_BE_CONFIGURED在本地运行时修改为本地数据集路径，在tinymind上运行时使用了2.1中数据集的路径**，num_steps都设置的100，max_evals设置为1，**eval_input_reader的shuffer在本机设置为了true，上传至tinymind时设置为true**
2. 将课程提供的run.py、run.sh、reference.py拷贝到models/research目录下；
3. 根据是在本地运行，还是tinymind上运行，调整run.sh中的 **output_dir和dataset_dir路径配置**；
4. 按照作业指南，删除model下本次作业无关代码，在本机进行试训练，发现没有错误后，上传至github，并在tinymind上新建模型进行运行，注意调整以上 **粗体参数**,因本机性能的缘故，本机没有将训练完全执行完成。tinymind最后一次执行的链接如下：https://www.tinymind.com/executions/b8um3dy5

以下给出tinymind运行的的log截图：
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_week9_run_log.png"/></div>

### 2.3执行的输出
1. 执行的输出连接：https://www.tinymind.com/executions/b8um3dy5/output

以下给出tinymind运行的的output截图：
<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_week9_run_output.png"/></div>

2. 在实际的tinymind训练过程中，因为参数设置的不准确，走了不少弯路，其中有个使用了不合理数据集的模型，地址：https://www.tinymind.com/SDMrFeng252/quiz-w9-ssd-mobilenet

## 3.学习心得
1. 本周课程录播内容听了至少两遍，每个细节基本都注意到，并且做了详细的笔记。但是在做作业时，还是感觉比较吃力，浪费了不少时间。
2. 不清楚实际工作中图片检测的应用场景和难度，感觉本周作业仅仅算是一个demo，自己掉包侠的工作都未做好。

PS:作业做完已三天，受限于催人心智的牙疼，作业总结迟迟未能完成，好好准备10月20号的线下实践，希望可以收获些自信。
