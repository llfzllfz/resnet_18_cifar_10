# resnet_18_cifar_10

利用resnet-18实现对cifar-10数据集的分类

文件目录

---config.py

---data.py

---network.py

---train.py

## config.py

参数设置

---batch_size 批大小

---path 数据存放文件(建议采用相对路径)

---model_path 模型存放路径

---model_name 模型名称

---epoh 训练的轮数

## data.py

文件的预处理

支持批读取文件

## network.py

神经网络的构建

采用resnet-18的方式构建

## train.py

训练模型
