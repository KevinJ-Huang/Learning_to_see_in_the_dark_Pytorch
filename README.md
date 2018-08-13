# Learning_to_see_in_the_dark_Pytorch

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

直接运行squid里的train.py就可以了

另说明：还要有一个models文件用来存放模型，dataset里还要有一个Sony文件存放数据，里面子文件(long/short)存放长短曝光数据
