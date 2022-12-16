# ADACRS
这是对于高校合作论文 **Towards Hierarchical Policy Learning for Conversational Recommendation with Reinforcement Learning** 的代码实现.

已基于[YAPF](https://github.com/google/yapf)(Yet Another Python Formatter)的pep8规则规范化代码

## 环境需求
python==3.7

环境所需包在```requirements.txt```

## 数据准备
开源数据集部分，使用两个基于公开的对话数据集**Yelp**和**LastFM**处理之后的用于CRS的数据

关于这两个数据集以及 **Last_FM*** 与 **Yelp*** 的描述，请参考[SCPR的数据介绍](https://cpr-conv-rec.github.io/manual.html/#environment-requirement)

下载链接：[google drive](https://drive.google.com/file/d/1H75SOzOkps4fYeu9A0qF7voDkPw8cMhE/view?usp=sharing)

### 1.数据解析
解压```dataset.zip```文件，并将其置于```adacrs/```路径下
```
unzip dataset.zip
mv ./dataset/ adacrs/
```

包括user，item和attribute的信息


### 2.处理dataset与kg
首先参照```utils/utils_load_save.py```中```TMP_DIR```以及```DATA_DIR```指定```scripts/preprocess.sh```中的```ADACRS_DIR```
```
bash scripts/preprocess.sh
```

将```DATA_DIR```中的数据处理为对应的dataset.pkl及kg.pkl到```TMP_DIR```中

### 3.加载graph embeddings
使用[OpenKE](https://github.com/thunlp/OpenKE)中的TransE方法训练Graph的embeddings，将训练好的embeddings放置于```adacrs/TMP_DIR/embeds/```路径下


## 训练与测试
### 1.训练代码
训练模型
```
bash scripts/train.sh
```

#### 重要调参的参数意义
可以通过```python train.py -h```来查看更多的参数细节

- ```seed```：随机种子
- ```batch_size```：训练batch size大小
- ```gamma```：extrinsic reward的系数
- ```memory_size```：强化学习时储存memory的大小
- ```data_name```：数据集名称，```{LAST, LAST_STAR, YELP, YELP_STAR}```其中的一个
- ```max_turn```：最大交互轮数,即T
- ```max_steps```：最大训练步数
- ```attr_num```：attribute的数量
- ```ask_num```：每一轮询问attribute的数量
- ```eval_num```：每```eval_num```测试一次模型metric
- ```save_num```：每```save_num```保存一次模型测试结果
- ```observe_num```：每```observe_num```输出一次metric结果
- ```embed```：graph embedding类型, 默认使用TransE
- ```gcn```：是否使用gcn
- ```seq```：gcn中sequential encoder选择

### 2.结果存储
训练结束后，对应指标结果将被存放在```TMP_DIR/RL-log-merge/```路径下。






