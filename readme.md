## 通用模块 

### loss

每个文件一种loss，每个loss之间是平等独立的， 拷走即用

### net

每个文件一种net结构，每个net结构之间是平等独立的， 拷走即用

### model

每个文件是一个model，它组合了各种loss，各种net, 定义了拟合和预测等接口. 每个model之间是平等独立的， 拷走即用

### data

每个文件一个dataset处理, 每个dataset之间平等独立，拷走即用 
每种dataset的预处理和后处理(例如outmask的转换)都在单个文件中，例如:分割dataset，预处理数据集可直接调用 data/seg_data.py  src_dir dst_dir 
这样某个dataset相关的所有操作都在单个文件中。

### utils

可视化, log等基础模块 


## 使用:

### step1: 定义config文件

定义训练测试流程的迭代数，数据集路径，model的插拔, dataloder的插拔

### step2: 执行train.py 

例如: python -m squid.train  experiments/sr_config.py  

读取config,  驱动train, validate 这2种主流程, 同时支持snapshot断点, 用若干样本对模型效果进行监控 等功能


### step3: inference.py 

inference.py  模型评测

##  mnist example:

python -m squid.train experiments/mnist/config.py
