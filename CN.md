[中文](./CN.md)  /  [English](./README.md)

### 1.基于多分支注意力改进的卷积神经网络算法（MBAA-CNN）

![](https://s2.loli.net/2024/01/04/ZBG6MHalDo5pUnj.png)

| Method   | 未标注 | 标注   |
| -------- | ------ | ------ |
| SVM      | 0.6981 | 0.7020 |
| RNN      | 0.8078 | 0.8104 |
| KNN      | 0.8567 | 0.8574 |
| CNN      | 0.9097 | 0.9187 |
| MBAA-CNN | 0.9897 | 0.9909 |



## 2.部署方法建议

- 训练

```python
python main.py
```

- 部署

```python
python Model_Apply.py
```



- #### 提供测试使用预处理后的[十种细菌拉曼光谱test文件](https://drive.google.com/file/d/1WeH_uRzx1HT1DCyYilERKbZkCHOnwRav/view?usp=drive_link)，以及我们预训练好的[分类模型](https://drive.google.com/file/d/12Q4Vd-eN2-rNCBofm0dYQdozMhqTJg34/view?usp=drive_link)。使用时，请将test文件解压到Final_Data文件夹中并在Model_Apply.py中修改文件路径。对于原数据标签可以自行设置字典添加，例如：

```
['Cns', 'E. cloacae', 'E. coli', 'K. pneumoniae', 'MC', 'MRSA', 'MSSA', 'P. aeruginosa', 'P. vulgaris', 'S. epidermidi']
```

- 推荐使用Linux环境创建虚拟环境进行训练





使用请标明出处引用！

#### **参考：**

```
[1] J. Feng et al., "Attention Multibranch Convolutional Neural Network for Hyperspectral Image Classification Based on Adaptive Region Search," in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 6, pp. 5054-5070, June 2021, doi: 10.1109/TGRS.2020.3011943.

[2] Wang, Hongtao & Xu, Linfeng & Bezerianos, Anastasios & Chen, Chuangquan & Zhang, Zhiguo. (2020). Linking Attention-Based Multiscale CNN with Dynamical GCN for Driving Fatigue Detection. IEEE Transactions on Instrumentation and Measurement. 57. 1-1. 10.1109/TIM.2020.3047502. 

[3] Xunli Fan, Shixi Shan, Xianjun Li, Jinhang Li, Jizong Mi, Jian Yang, Yongqin Zhang,
Attention-modulated multi-branch convolutional neural networks for neonatal brain tissue segmentation,Computers in Biology and Medicine,Volume 146,2022,105522,ISSN 0010-4825,https://doi.org/10.1016/j.compbiomed.2022.105522.(这个论文没有下载入口，只参考了引言)

[4] H. Zhang, Y . Li, Y . Zhang, and Q. Shen, “Spectral-spatial classification
of hyperspectral imagery using a dual-channel convolutional neural
network,” Remote Sens. Lett., vol. 8, no. 5, pp. 438–447, May 2017.

[5] Y . Chen, H. Jiang, C. Li, X. Jia, and P. Ghamisi, “Deep feature extrac-
tion and classification of hyperspectral images based on convolutional
neural networks,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 10,
pp. 6232–6251, Oct. 2016.

[6] Y . Xu, L. Zhang, B. Du, and F. Zhang, “Spectral–spatial unified networks
for hyperspectral image classification,” IEEE Trans. Geosci. Remote
Sens., vol. 56, no. 10, pp. 5893–5909, Oct. 2018.

[7] Yin, Wenpeng, et al. "Comparative study of CNN and RNN for natural language processing." arXiv preprint arXiv:1702.01923 (2017).

[8] Visin, Francesco, et al. "Renet: A recurrent neural network based alternative to convolutional networks." arXiv preprint arXiv:1505.00393 (2015).

[9] Moldagulova, Aiman, and Rosnafisah Bte Sulaiman. "Using KNN algorithm for classification of textual documents." 2017 8th international conference on information technology (ICIT). IEEE, 2017.

[10] Chauhan, Rahul, Kamal Kumar Ghanshala, and R. C. Joshi. "Convolutional neural network (CNN) for image detection and recognition." 2018 first international conference on secure cyber computing and communication (ICSCCC). IEEE, 2018.

[11] Jakkula, Vikramaditya. "Tutorial on support vector machine (svm)." School of EECS, Washington State University 37.2.5 (2006): 3.

```

