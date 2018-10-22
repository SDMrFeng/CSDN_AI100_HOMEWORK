# 第十一周作业：使用tensorflow中的rnn相关操作，以作业提供的《全宋词》为训练数据，训练一个人工智能写词机
**学员：** 冯学智

**手机：** 18210239929

**微信群昵称：** 北京-开发-冯学智

## 作业 1.Word embedding
**数据集:**
本作业使用QuanSongCi.txt文件作为数据集，含空格在内1903072个字符，。
- 全宋词资料不同于英文，不使用分词，直接将每个单字符作为一个word，将整个文本转为word序列即可。
- 全宋词全文，含空格在内共1903072个字符，6010个不同的单字符，这里只取出现次数最多的前5000个单字符（频次在5000名之后的归为UNK）。

**代码**

参考代码：
https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

实际对全宋词进行训练，生成embedding的实现代码：（主要改动在第一步数据读取时）
https://github.com/SDMrFeng/quiz-w11-rnn/blob/master/Word2Vec.py

**训练小记**
1. 参考代码和录播课程中讲解Word2Vec使用的是同一基础的代码，在听课过程中即完成了随堂复现，所以代码部分没有太多障碍；
2. matplotlib绘图时，对中文均以方框代替，在解决过程中遇到了一些问题，根据网上的参考方案，增加了matplotlib下的simhei字体。（本机除了tensorflow环境还有python2和python3的环境，在本次作业中使用python3，也提高了动手配置环境的能力）
3. 按照作业要求，在代码中增加了word和index对应dictionary和reverse_dictionary的保存操作，增加了结果embedding.npy的保存操作。

**训练结果**

按照作业提示，设置了400,000个step，生成了dictionary.json(78K)、reversed_dictionary.json(88K)和embedding.npy(2.6M)三个文件，将训练结果降维后的输出图片如下：
图中圈出几处明显同类词的聚集范围：
- 金、银、宝、玉
- 一、二、三、四、五、六、八、九、十、百、千、万、几、半、数
- 小、浅、深、静、轻、微、薄、浓、淡、凝、多
- 言、语、声、闻、听、看、吟
- 独、孤、疏、断、残
- 垂、低、远、高
- 东、南、西、北
- 花、叶、林、树、枝
- 标点、。，）（集中在一起
- UNK字符位置也相对独立

<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_quiz_week11_output_word2vec.png"/></div>

**心得体会**

代码理解：
1. 准备数据：案例中和录播课程中使用的是从网上下载的text8.zip中的全英文单词的文本，作业中直接读取了QuanSongCi.txt里的中文文本。
2. 建立word与index的对应dictonary和reverse_dictionary：首先对word序列使用collections.Counter().most_common()函数进行了排序，取出频次最高的前4999个word和count对来填充字典，然后遍历所有word，将未出现在前4999的word计数并作为UNK字符加到字典中。
3. 构建生成训练data和lable的函数，关键是理解滑窗的逻辑。其中训练数据和label均是以word的index表示的。
4. 构建训练的计算图，将5000个不同word塞入到128维空间。重点理解skip-gram和cbow的区别、embedding_lookup原理、把多分类问题转为二分类优化计算的nce_loss、利用余弦距离计算相识度。
5. 使用了作业建议的400,000个step，最后输出训练结果embedding文件
6. 利用pca降维，使用matplotlib绘制图形化结果。

Word2Vec理解：
1. Word2Vec是Word Embedding的一种实现，将word表示由高维稀疏转为低维稠密(相对高维而言)。
2. OneHot Representation表达效率太低，词间关联没有体现，在使用中计算量过大。Word Embedding是Distributed Representaion，很好的解决了这方面的问题，尤其Word2Vec同类词的"对应关系"，可以用同向同大小的向量表示。
3. cbow和skip-gram都是在word2vec中用于将文本进行向量表示的实现方法。在cbow方法中，是用周围词预测中心词，从而利用中心词的预测结果情况，使用GradientDesent方法，不断的去调整周围词的向量。当训练完成之后，每个词都会作为中心词，把周围词的词向量进行了调整，这样也就获得了整个文本里面所有词的词向量。skip-gram是用中心词来预测周围的词。在skip-gram中，会利用周围的词的预测结果情况，使用GradientDecent来不断的调整中心词的词向量，最终所有的文本遍历完毕之后，也就得到了文本所有词的词向量。
4. Word2Vec的语言模型有两种对传统的神经网络改进的训练方法：一种是基于Hierarchical Softmax的（主要是时霍夫曼树，以前用到时主要是做压缩数据使用），另一种是基于Negative Sampling的。

## 作业 2.RNN训练
**数据集:**

没有再从新生成训练数据，而是直接使用作业1中生成的匹配的dictionary.json、reversed_dictionary.json和embedding.npy作为本部分的数据。

**代码**

使用了本周作业https://gitee.com/ai100/quiz-w10-code 的基础代码并进行了响应填充，完整代码见：
https://github.com/SDMrFeng/quiz-w11-rnn

其中utils.py的修改内容如下：
```python
def get_train_data(vocabulary, batch_size, num_steps):
    ################# My code here ###################
    # 有时不能整除，需要截掉一些字
    data_partition_size = len(vocabulary) // batch_size
    word_valid_count = batch_size * data_partition_size
    vocabulary_valid = np.array(vocabulary[: word_valid_count])
    word_x = vocabulary_valid.reshape([batch_size, data_partition_size])

    # 随机一个起始位置
    start_idx = random.randint(0, 16)
    while True:
        # 因为训练中要使用的是字和label(下一个字)的index，
        # 但这里没有dictionary，无法得到index
        # 所以将每个time step返回的训练数据长度是num_steps+1
        #     其中前num_steps个字用于训练(训练时转化为index)
        #     从第2个字起的num_steps个字用于训练的label(训练时转化为index)
        if start_idx + num_steps + 1 > word_valid_count:
            break

        yield word_x[:, start_idx: start_idx + num_steps + 1]
        start_idx += num_steps
    ##################################################
```

model.py的修改内容如下：
```python
with tf.variable_scope('rnn'):
    ################# My code here ###################
    cells = [tf.nn.rnn_cell.DropoutWrapper(
                          tf.nn.rnn_cell.BasicLSTMCell(self.dim_embedding),
                          output_keep_prob=self.keep_prob
                          ) for i in range(self.rnn_layers)]

    rnn_multi = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    self.state_tensor = rnn_multi.zero_state(self.batch_size, tf.float32)

    outputs_tensor, self.outputs_state_tensor = tf.nn.dynamic_rnn(
        rnn_multi, data, initial_state = self.state_tensor, dtype=tf.float32)

    tf.summary.histogram('outputs_state_tensor', self.outputs_state_tensor)

seq_output = tf.concat(outputs_tensor, 1)
    ##################################################

# flatten it
seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])

with tf.variable_scope('softmax'):
    ################# My code here ###################
    softmax_w = tf.Variable(tf.truncated_normal(
                [self.dim_embedding, self.num_words], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(self.num_words))

    tf.summary.histogram('softmax_w', softmax_w)
    tf.summary.histogram('softmax_b', softmax_b)

logits = tf.reshape(tf.matmul(seq_output_final, softmax_w) + softmax_b,
                  [self.batch_size, -1, self.num_words])
    ##################################################

tf.summary.histogram('logits', logits)

self.predictions = tf.nn.softmax(logits, name='predictions')

#print('logits shape:', logits.get_shape())
#print('labels shape:', self.Y.get_shape())
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=self.Y)
self.loss = tf.reduce_mean(loss)
tf.summary.scalar('logits_loss', self.loss)
```

train.py的修改内容如下：
```python
for dl in utils.get_train_data(vocabulary,
                  batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
    ################# My code here ###################
    # dl(data+label), dl.shape[0]是batch_size, shape[1]是num_steps+1，
    # 内容都是字，需要转为index
    # 注意：其中前num_steps个字用于训练(训练时转化为index)
    #      从第2个字起的num_steps个字用于训练的label(训练时转化为index)
    dl_word_len = dl.shape[1]
    dl_index = utils.index_data(dl, dictionary)  # 找到每一个字对应的index
    d_index = dl_index[:, :dl_word_len - 1]
    l_index = dl_index[:, 1:]
    feed_dict = {model.X: d_index, model.Y: l_index,
                  model.state_tensor: state, model.keep_prob: 0.5}
    ##################################################

    gs, _, state, l, summary_string = sess.run(
        [model.global_step, model.optimizer, model.outputs_state_tensor,
         model.loss, model.merged_summary_op], feed_dict=feed_dict)
    summary_string_writer.add_summary(summary_string, gs)
```

**训练小记**
1. 录播课程中讲解RNN和LSTM实现的代码，在听课过程中即完成了随堂复现（没找到直接文件的，都是手敲了一遍，加深了理解），所以代码部分问题不多，主要是作业基础代码中训练data和label传递时需要注意一些细节；
2. 在本机测试train.py文件执行没有问题后，没对sample.py和train_eval.py进行细节关注，上传到tinymind进行训练了两个多少小时，发现sample.py报错了。运行连接：（手动kill了）
https://www.tinymind.com/executions/aktuxj3n
3. 仔细分析问题，发现sample.py中希望得到的时二阶矩阵，而自己实现时输出的logits是三阶矩阵，对sample.py文件修改了两行代码，在本地测试运行通过后，上传tinymind进行了第二次运行：
https://www.tinymind.com/executions/bm78i4nr

**训练结果**

训练的log中没有出现作业提示的文字转码问题，正常输出了文字，最后的log截图如下：（可以明确看到，RNN学会了标点的使用，记住了一些词牌的名字）

<div align=center><img  src="https://raw.githubusercontent.com/SDMrFeng/photosets/master/csdn_quiz_week11_output_rnn_log.png"/></div>

**心得体会**

代码理解：
1. 梯度剪裁操作：tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)，将梯度强行控制在(-5,5)区间，解决可能出现的梯度爆炸问题。
2. 录播课程代码讲解和作业中比较需要细心的地方是训练数据和lable的构造，都是以word的一下个word作为label。这样的效果就是在验证时，取一个word就会得到下一个word,类似背诵文章。

RNN和LSTM理解：
1. 相对于CNN，主要区别是：即时输入，存储之前的输入信息；
2. 结构简单，每个单元只有一个隐层，信息抽取能力弱，但增加了时间轴，在时间轴上形成了深度网络，每层共享权重。实际在处理文本时，是在输入数据上进行滑动，保证时序的连续性，来模拟即时输入。而CNN中划分数据时较为随机。
3. 反向传播包括layer间的传递和time_step间的传递，不是每处理一个输入数据就反向传播一次，而是经过几个time_step之后，统一计算loss之后，进行一次反向传播操作。
4. tensorflow的static_rnn实现时，会生成整个的计算图，且对输入的维度timesteps有固定要求；dynamic_rnn实现时，不真正生成整个计算图，而是在计算单元cell外包裹一个loop，对timesteps长度无固定要求，比较符合实际自然语言情况。
5. LSTM中增加了三个gate，分别控制要遗忘多少原来记住的信息、要记住多少当前输入的信息、要将多少信息输出到下一时刻。当然LSTM的“权重”涉及到的参数数量也增长了，变为普通rnn的4倍左右。

## 随堂作业部分

1. RNN的反向传播的理解和代码补充实现，做的时候稍微费了些时间，但第一次运行就输出了正确结果，理解了会在模型层次和时间层次两个方向进行反向传播，就可以比较轻松的完成： https://github.com/SDMrFeng/quiz-w11-rnn/blob/master/BPTT-Test.ipynb
2. 随堂练习（调整参数，观察模型学习到了数字排列规则的情况）调整number_steps和state_size，根据loss变化判断模型是否学习到了01数字排列规则，学习到了一个规则还是两个规则：（用了20组左右的参数，观察得到了一些体会）
- number_steps太小时，什么都记不住，loss会出现震荡；
- number_steps增大的同时，也许保证epoch_nums，即保证BP次数，否则loss下降不到最低点，达不到最优效果；
- state_size太小时，记住的规则有限，最多学到一个规则。

## 本周总结
用一周的时间讲解了RNN入门，内容压缩厉害，理解随堂代码讲解、完成本周作业需要花费时间较多，恰逢十一假期停课，否则一周工作之外的时间很难完成。现在完成了作业，也不清楚离NLP实战还有多远，哈哈。

结束了录播课程的学习，感觉前期的内容有些已经遗忘了，对于换份机器学习方面的工作心里依旧很没底，复习之后，希望在实战阶段能多拾起来一些、再学习一些新知识。作业虽不是目的，做好作业是对自己学习的一种验证，拿出态度、写好作业总结，也是对批改作业老师的一种尊重。感谢老师一直以来的认真、细致的审阅。
