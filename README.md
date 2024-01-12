[中文](./CN.md)  /  [English](./README.md)

# 1.Implementation process

## 1.1Architectural model

### 1.1.1Convolutional neural network algorithm based on multi-branch attention improvement (MBAA-CNN)

<img src="https://s2.loli.net/2024/01/09/MzOkCBdHSRpAtIQ.png" style="zoom:67%;" />	



| Method   | UNLABEL | LABEL  |
| -------- | ------- | ------ |
| SVM      | 0.6981  | 0.7020 |
| RNN      | 0.8078  | 0.8104 |
| KNN      | 0.8567  | 0.8574 |
| CNN      | 0.9097  | 0.9187 |
| MBAA-CNN | 0.9897  | 0.9909 |



## 2. Deployment method suggestions

- Training

```python
python main.py
```

- Deployment

```python
python Model_Apply.py
```



- #### Provide preprocessed [Ten bacterial Raman spectrum test files](https://drive.google.com/file/d/1WeH_uRzx1HT1DCyYilERKbZkCHOnwRav/view?usp=drive_link) for testing, and our pretrained [Classification model ](https://drive.google.com/file/d/12Q4Vd-eN2-rNCBofm0dYQdozMhqTJg34/view?usp=drive_link). When using it, please extract the test file to the Final_Data folder and modify the file path in Model_Apply.py. For original data labels, you can set up a dictionary yourself, for example:

```
['Cns', 'E. cloacae', 'E. coli', 'K. pneumoniae', 'MC', 'MRSA', 'MSSA', 'P. aeruginosa', 'P. vulgaris', 'S. epidermidi']
```

- It is recommended to use a Linux environment to create a virtual environment for training



**Please indicate the source for use!**

#### **References:**

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

