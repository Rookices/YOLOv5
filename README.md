本工程在https://github.com/ultralytics/yolov5/releases/tag/v4.0 的基础上，替换部分网络结构使其适配海思平台；添加数据集格式化脚本、通道剪枝模块及模型转换模块

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
│       ├── train
│       ├── val
│       └── test
└── others
```

----

##### 3.配置文件

1. 超参(--hyp)： ./data/hyp.scratch.yaml   (可自行修改超参，如学习率，数据增强所用的超参等)
2. 模型配置(--cfg)： ./models/yolov5s.yaml  (为方便海思移植，已将focus层替换为conv层，upsample层替换为deconv层，且deconv层支持yolo缩放)
3. 数据集(--data)： ./data/ab.yaml   (根据自己数据集**更改图片路径**，类别及类别数)
4. 其他模型保存策略(--other_save)：可打开包含基于最优F1以及Wrecall的模型保存方式
P.S. 在训练和测试阶段，无特殊情况，配置文件均为上述默认设置。
---

##### 4.训练（含剪枝：步骤2-4）

1. 原始模型训练
   ```
   python train.py --weights ./weights/yolov5s.pt --epochs 1000 
   ```
   
2. 稀疏训练，训练次数以能恢复原模型精度为准
   ```
   python train.py --sl_factor <设定稀疏率，建议6e-4> --weights <半精度模型原始模型> --epochs 500
   ```
   
3. 剪枝，剪枝模型路径下会自动生成剪枝稀疏度分布图W
   ```
   python pruning.py --weights <稀疏后模型> --save_path <剪枝模型保存路径> --thres <默认剪枝阈值0.01>
   ```
   
4. 模型微调
   
   ```
   python train.py --weights <剪枝后模型> --epochs 500 --ft_pruned
   ```
   
5. 知识蒸馏

   ```
   python train_di.py --weights <Student模型> --t_weights <Teacher模型> --distill --temperature 4 --dist_loss kl --epochs 500 
   ```
---

##### 5.测试

1. 图片推理
   ```
   python detect.py --source <图片路径> --weights <模型> --conf-thres <置信度阈值默认0.25> --iou-thres <IoU阈值默认0.5>
   ```
2. 模型评估(含FLOPs及耗时)
   ``` 
   python test.py --source <图片路径> --weights <模型>  --conf-thres <置信度阈值默认0.25> --iou-thres <IoU阈值默认0.5> --task <选择数据集val or test>
   ```
   P.S. 更改ab.yaml中的test路径可实现测试集更换
   
3. 批量导出稀疏度分布图(自动识别last前缀模型，导出路径为模型根目录)
   ```
   python img_bn_analyze.py --weights <模型所在根目录>
   ```
---

##### 6.自动计算

1. 生成真实标签
   ```
   python get_gt_txt.py --testPath <测试集图片路径>
   ```
2. 生成预测标签. 即在第5.1基础上将--save-txt, --save-conf 设置为True   
   ```
   python detect.py --source <图片路径> --weights <模型> --conf-thres <置信度阈值默认0.25> --iou-thres <IoU阈值默认0.5> --save-txt --save-conf
   ```
3. 自动计算，结果保存在result文件夹下
   ```
   python auto_analysis.py --testPath <测试集图片路径>
   ```
---

##### 7.模型转换(支持onnx及caffe)  

1. yolo2onnx

   python models/export.py --weights <需转换模型> --img-size <训练时图片输入尺寸>

2. onnx简化
   
   python -m onnxsim <onnx模型> <简化模型保存路径>
   
3. onnx2caffe   
   
   1）修改 ./yolov5_onnxcaffe/convertCaffe.py
   ```
        onnx_path        <onnx简化模型>
        prototxt_path    <prototxt保存路径>
        caffemodel_path  <caffemodel保存路径>
   ```

   2）python yolov5_onnxcaffe/convertCaffe.py
---

##### todo list
- [x] 训练样本txt 要支持多个，改成ssd工程类似，测试txt要能指定
- [x] val集 划分不合理，现在的方式，单个txt的尾部数据全部到了val集。例如multicode
- [x] detect.py 中load 测试图片函数统一放到load_image.py