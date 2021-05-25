本工程在https://github.com/ultralytics/yolov5/releases/tag/v4.0的基础上，添加数据集格式化代码及基于海思平台修改部分网络结构

----

##### 1.环境配置

Python>=3.6 and pytorch>=1.7 (原作者是要求python>=3.8, 我们在python=3.6.9也可以正常运行)

pip install -r requirements.txt

------------------

##### 2.数据集制作

支持txt图库下载，由参数args.urlPath传入，并完成数据集划分及格式转换，第一次下载图片时查看路径下是否有之前下载的图片，如不需要则删除，只第一次训练时需下载图片。默认训练时加载1.txt，测试时加载2.txt。  
数据集树状结构为

```
yolov5
├── data   
│   ├── images   存放图片
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── labels   存放标签
│   │   ├── train
│   │   ├── val
│   │   └── test
└── others
```

----

##### 3.配置文件

1. 超参： ./data/hyp.scratch.yaml   (可自行修改超参，如学习率，数据增强所用的超参等)
2. 模型配置： ./models/yolov5s.yaml  (可根据所需更改模型结构，例如海思芯片不支持Focus结构，可更改Focus为一个卷积层)
3. 数据集： ./data/ab.yaml   (根据自己数据集**更改图片路径**，类别及类别数)

---

##### 4.训练

./weights 下载预训练模型保存到该路径，考虑到模型大小，目前只下载 yolov5s.pt

python train.py <font color=#0099ff>--weights</font> ./weights/yolov5s.pt <font color=#0099ff>--cfg</font> ./models/yolov5s.yaml <font color=#0099ff>--data</font> ./data/ab.yaml <font color=#0099ff>--epochs</font> 1000 

1）稀疏训练

python train.py --sl_factor 6e-4

2）剪枝训练

python pruning.py  --thres 0.01

3）模型微调

python train.py --ft_pruned

---

##### 5.测试

python detect.py <font color=#0099ff> --source </font> 图片路径 <font color=#0099ff> --weights</font> 训练后保存的模型  <font color=#0099ff>--conf</font> 0.25

---

##### 6.自动计算
1. 运行 get_gt_txt.py 生成真实标签.(需传入参数--testPath 测试机图片路径,即跟detect.py 中的 --source 一样) 
2. 生成预测标签. 即在第五步测试时，**--save-txt, --save-conf 要设置为True** 即：   
python detect.py <font color=#0099ff> --source </font> 图片路径 <font color=#0099ff> --weights</font> 训练后保存的模型  <font color=#0099ff>--conf</font> 0.25 --save-txt --save-conf
3. 运行 auto_analysis.py 结果保存于result文件夹下（需传入参数--testPath 测试机图片路径,即跟detect.py 中的 --source 一样）。   
在测试阶段 detect.py --source 设置测试图片路径，而get_ge_txt.py --testPath 需定位到测试图片路径。默认为'./test/img'  
---

##### 7.模型转换   
模型转换[参考](https://github.com/Wulingtian/yolov5_caffe)   
为方便海思移植，修改网络结构  
把yolov5s.yaml的focus层替换为conv层（stride为2），upsample层替换为deconv层,修改如下所示：   



    backbone:
          # [from, number, module, args]    
          #Fous层[[-1, 1, Focus, [64, 3]],  # 0-P1/2   
           [[-1, 1, conv, [128, 3, 2]],  # 0-P1/2   
           [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4   
           [-1, 3, C3, [128]],   
           [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8   
           [-1, 9, C3, [256]],   
           [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16   
           [-1, 9, C3, [512]],   
           [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32   
           [-1, 1, SPP, [1024, [5, 9, 13]]],   
           [-1, 3, C3, [1024, False]],  # 9   
          ]
    
    head:
        [[-1, 1, Conv, [512, 1, 1]],   
        #上采样[-1, 1, nn.Upsample, [None, 2, 'nearest']],   
        [-1, 1, nn.ConvTranspose2d, [256, 256, 2, 2]],   
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4   
        [-1, 3, C3, [512, False]],  # 13   
    
        [-1, 1, Conv, [256, 1, 1]],   
        #上采样[-1, 1, nn.Upsample, [None, 2, 'nearest']],   
        [-1, 1, nn.ConvTranspose2d, [128, 128, 2, 2]],   
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3   
        [-1, 3, C3, [256, False]],  # 17 (P3/8-small)    
    
        [-1, 1, Conv, [256, 3, 2]],   
        [[-1, 14], 1, Concat, [1]],  # cat head P4   
        [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)   
    
        [-1, 1, Conv, [512, 3, 2]],   
        [[-1, 10], 1, Concat, [1]],  # cat head P5   
        [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)   
    
        [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)   
        ]

 1. 模型转换 YOLO→onnx
    - 安装onnx和onnx-simplifier

    ```bash
    pip install onnx
    pip install onnx-simplifier
    ```
    - 修改配置

        修改models/export.py下**opset_version=10**  原为12

    - 模型转换

        - python models/export.py --weights 训练得到的模型权重路径 --img-size 训练图片输入尺寸

      ```bash
      python models/export.py --weights ./weights/yolov5s_helmet.pt
      ```

    - 模型简化

        - python -m onnxsim 模型名称 yolov5s-simple.onnx（得到最终简化后的onnx模型）

      ```bash
      python -m onnxsim ./weights/yolov5s.onnx yolov5s_helmet_simple.onnx
      ```
    
 2. 模型转换 onnx→caffe   
    
    - 确保已经搭建好caffe 
    - 设置路径   
        ```bash
        cd yolov5_onnx2caffe
        ```
    - 修改 convertCaffe.py 中路径   
      
          设置onnx_path（上面转换得到的简化后onnx模型），prototxt_path（caffe的prototxt保存路径），caffemodel_path（caffe的caffemodel保存路径）
     - 转换 
        ```bash
        python convertCaffe.py
        ```
       得到转换后的caffemodel.  
    
    

3. todo list
- [x] 训练样本txt 要支持多个，改成ssd工程类似，测试txt要能指定
- [x] val集 划分不合理，现在的方式，单个txt的尾部数据全部到了val集。例如multicode
- [x] detect.py 中load 测试图片函数统一放到load_image.py