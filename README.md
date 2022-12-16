# ADACRS
这是对于高校合作论文 **Towards Hierarchical Policy Learning for Conversational Recommendation with Reinforcement Learning** 的代码实现.

已基于[YAPF](https://github.com/google/yapf)(Yet Another Python Formatter)的pep8规则规范化代码

## 环境需求
python==3.7

环境所需包在```requirements.txt```

## 数据准备
开源数据集部分，使用两个基于公开的对话数据集**Yelp**和**LastFM**处理之后的用于CRS的数据

关于这两个数据集以及 **Last_FM^*^** 与 **Yelp^*^** 的描述，请参考[数据介绍](https://yuque.antfin.com/docs/share/3b425aef-6b4e-4161-91a4-86b4d544899c?#%20%E3%80%8A%E8%9E%8D%E5%90%88%E4%B8%93%E5%AE%B6%E7%BB%8F%E9%AA%8C%E7%9A%84%E5%AF%B9%E8%AF%9D-%E6%95%B0%E6%8D%AE%E5%8F%8A%E5%8F%82%E8%80%83%E3%80%8B)或[SCPR的数据介绍](https://cpr-conv-rec.github.io/manual.html/#environment-requirement)

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

基础DQN训练流程：
1. 用随机的网络参数$\omega$初始化$Q_{\omega}(s,a)$网络 
2. 复制相同的参数$\omega^- \gets \omega$来初始化目标网络$Q_{\omega^-}$ 
3. 初始化经验回放池子$R$ 
4. **for** 序列$e=1 \rightarrow E$ **do** ```Trainer```
&ensp; &ensp; 获取环境初始状态$s_1$ ```Env```中实现
&ensp; &ensp; **for** 时间步 $t=1 \rightarrow T$ **do** ```Trainer```
&ensp; &ensp; &ensp; &ensp; 根据当前网络$Q_{\omega}(s,a)$以贪婪策略选择动作$a_t$ ```select_action function```
&ensp; &ensp; &ensp; &ensp; 执行动作$a_t$，获得奖励$r_t$，环境状态变成$s_{t+1}$ 
&ensp; &ensp; &ensp; &ensp; 把$(s_t, a_t, r_t, s_{t+1})$储存到回放池$R$中 
&ensp; &ensp; &ensp; &ensp; $R$中数据足够后，从$R$中采样$N$个数据$\{(s_i, a_i, r_i, s_{i+1})\}_{i=1,2,...,N}$ 
&ensp; &ensp; &ensp; &ensp; 对每个数据，用目标网络计算$y_i = r_i + \gamma max_{a} Q_{\omega^-}(s_{i+1},a)$ 
&ensp; &ensp; &ensp; &ensp; 最小化目标损失$L=\frac{1}{N} \sum_i(y_i-Q_{\omega}(s_i,a_i))^2$, 以此更新当前网络$Q_{\omega}$ ```policy.agent.learn```
&ensp; &ensp; &ensp; &ensp; 更新目标网络 ```policy.agent.update```
&ensp; &ensp; **end for**
**end for**



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






