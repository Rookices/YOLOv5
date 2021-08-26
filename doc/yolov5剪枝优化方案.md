# 剪枝优化方案汇总

## 剪枝技术路线 (**Update on Jun 10**)

《Rethinking the Value of Network Pruning》——剪枝留下的结构要比权重重要。剪枝的目的是获取简化的网络结构，而不是模型本身。

《The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks》——MIT研究员提出的彩票假设：剪枝后能恢复精度的模型，称为中奖网络，而迭代剪枝更容易获取中奖网络。

《One Ticket to Win Them All: Generalizing Lottery Ticket Initializations Across Datasets and Optimizers》——facebook团队针对中奖网络分别从迁移实验和数学推理，对其有效性以及泛化能力进行验证，得到以下结论：

小模型大数据集（重新初始化）>小模型小数据集（剪枝模型）≈大模型大数据集（预训练模型）

综上，目前剪枝工作是具备一定迁移性的，即迭代剪枝之后获得的网络结构能较好地迁移到相似(分类个数)分类任务。

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E4%BC%98%E5%8C%96%E6%96%B9%E6%A1%88-%E5%89%AA%E6%9E%9D%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF.png" alt="剪枝技术路线"/></center>

---

## 剪枝测试报告问题分析及初步优化方案 (**Update on Mar 27**)

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E4%BC%98%E5%8C%96%E6%96%B9%E6%A1%88-%E5%89%AA%E6%9E%9D%E4%BC%98%E5%8C%96%E6%96%B9%E6%A1%88.png" alt="剪枝优化方案"/></center>

#### 优化方案A 知识蒸馏

①概述 蒸馏就是输入两个模型：Teaher模型和Student模型，然后让Teacher模型去（离线）指导Student模型的训练。与微调的区别，微调时样本是严格one-hot的，Teacher模型可以给出连续的label分布，可以有效弥补监督信号不足。离线与在线的区别，离线指老师指导学生，在线指老师学生互助学习；自蒸馏：学生自己学。

②实现 输入两个模型，输出预期模型。整个方案的关键是如何定义损失函数：目前KL 和L2

③预期效果 得到的新模型继承Teacher模型的精度和Student模型的体积

---

#### 优化方案B 降低输入图片尺寸

①概述 降低输入图片尺寸会减少目标检测的面积，减少推理耗时。

②效果 精度有一定损失，速率X2，YOLOv5有3个尺寸anchor去检测物体，比如320，三个尺度分为别320/16*2=40,20,10。

③PC端测试结果，测试集为3000张图（0.25置信度+0.5IoU阈值）：条码漏检及精度下降导致map骤降。

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E4%BC%98%E5%8C%96%E6%96%B9%E6%A1%88-%E9%99%8D%E4%BD%8E%E8%BE%93%E5%85%A5%E5%B0%BA%E5%AF%B8%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C.jpg" alt="降低输入尺寸测试结果" style="zoom:100%;" /></center>

---

#### 优化方案C NAS理念

①概述 剪枝后通道数是参差不齐，不利于硬件平台加速。网络结构搜索NAS是由网络压缩延伸出来的新领域，让神经网络有目的的去构建网络结构，从而取代人工经验去完成网络结构的设计。

②实现 基于剪枝后的网络结构对通道补齐，向上补至8的倍数，重新进行训练。

③预期效果 修复板上精度及速度与PC的差距

④实际效果（PC端）：模型精度甚至高于original，推理时间进一步缩减

| 序号  |    模型    |           mAP@0.5            |     ▲      |   mmAP    | 模型大小  |             推理耗时             |  FLOPs   |
| :---: | :--------: | :--------------------------: | :--------: | :-------: | :-------: | :------------------------------: | :------: |
|   1   |  original  | <font color=red>0.965</font> |     0      |   0.850   |    15M    | **<font color=red>2.0ms</font>** |  17.5G   |
|   2   | sl449_0.08 |            0.962             |   -0.020   |   0.825   |   1.62M   |              1.0ms               |   3.8G   |
| **3** | **”nas“**  |          **0.982**           | **+0.017** | **0.877** | **1.16M** |          **0.7-0.8ms**           | **3.2G** |

---

#### 优化方案D 迭代剪枝

①概述 每次剪枝阈值较小，分多次剪枝。当前剪枝是一步到位，强行增大剪枝阈值会导致精度骤降。

②实现 在C的基础上人工定义网络结构，按照稀疏训练，剪枝，微调，重新定义网络结构的顺序循环，直到达到预期效果。

③预期效果 精度和推理速度得到一个更好地兼容值