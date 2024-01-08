[中文](./CN.md)  /  [English](./README.md)

# 1.Implementation process

## 1.1Architectural model

### 1.1.1Convolutional neural network algorithm based on multi-branch attention improvement (MBAA-CNN)

​		The MBAA-CNN architecture system is divided into two main stages: the first stage involves multi-branch attention network layers and non-local adaptive spatial segmentation strategies, and the second stage uses convolutional neural networks (CNN) to achieve multi-level spatial spectrum. Feature extraction. The diagram shows the application structure of MBAA-CNN in Raman spectrum data classification. In the preliminary stage, the preprocessed spectral data are non-locally adaptive segmented with the help of a peak detection algorithm to divide three independent data sections to lay the foundation for subsequent analysis. Subsequently, under the action of the multi-branch attention network layer embedded in MBAA-CNN, the system adaptively adjusts the weight of each segment data and dynamically integrates multi-branch features to prepare for the multi-layer input of CNN. In the second stage, the preliminary processed data is input into the shallow, middle and deep layers of each branch of the CNN, and undergoes a series of operations such as convolution, pooling, and dropout to extract comprehensive spatial-spectral features. Thereafter, the use of complementary information between different layers is maximized through the function of the Softmax layer. Finally, the main classifier uses the extracted features to complete the Raman spectrum classification of the test sample.

![](https://s2.loli.net/2024/01/08/VsJ8gCWvb4U9Rae.png)	

​																

​	

1. When we train the network, in order to make the model train faster and more accurately, we add an adaptive adjustment function of the learning rate, which can automatically adjust the learning rate according to the training data and the existing training volume, so that the training effect can be achieved Optimal.

   The specific model structure is as follows:

   1. We first divide the training data set into a training set and a validation set according to 4:1.
   2. Build MBAA-CNN network framework
   3. Input the training data into the MBAA-CNN network for 10,000 rounds of training
   4. After the model is trained, use the test data to test the model prediction results.
   5. Adjust the model parameters. After the model structure is optimized, test the final classification accuracy of the model and record the changes in the Loss value during training.



## 1.2 Experimental results of attention-based improved convolutional neural network (MBAA-CNN)

After training, this experiment randomly selected 50 Raman data of three types of bacteria for model evaluation.

### 1.2.1Unlabeled data mixture

### 1.2.2Labeled data mixture (10 types of bacterial training and classification effects)

##### 1.Changes in training accuracy

<img src="https://s2.loli.net/2024/01/08/OqRtmCT4leVPMUv.png" style="zoom:67%;" />

- The change in accuracy is ideal and meets the expected requirements:

  - After the 1500th round of training, the **training accuracy** of the model remains at around 98%
  - After the 3400th round of training, the **validation accuracy** of the model remains at around 95%

```
Model validation usually uses a data set independent of the training set and test set to evaluate model performance during the training process. It can be used to detect whether the model is overfitting or underfitting. If the model performs well on the training set but performs poorly on the validation set, it means the model may be overfitting. In this case, some methods can be taken, such as stopping training early or adding regularization, to prevent the model from overfitting.

Training accuracy represents the performance of the model on the current training data. After multiple rounds of training, the training accuracy will gradually increase, which indicates that the model has learned more data classification features. However, if the training accuracy starts to get very high, but the validation accuracy stops improving, this is a sign that the model is starting to overfit the training data.
```

##### 2.Changes in training LOSS value

<img src="https://s2.loli.net/2024/01/08/vpZ2zd5BibQtoCl.png" style="zoom:67%;" />

##### 3.The model classifies n types of bacterial data respectively.

<img src="https://s2.loli.net/2024/01/08/dIRhN5oP6bZfETX.png" alt="98.833%四种细菌分类情况Process" style="zoom: 33%;" />

<img src="https://s2.loli.net/2024/01/08/ZHFSP61TyM2Ju3b.png" alt="98.75%分类情况process" style="zoom: 40%;" />

##### 4.Simultaneous classification of ten kinds of bacteria

**（1）.UNLABEL**

<img src="https://s2.loli.net/2024/01/04/eCnNmoRU6cQVF1A.png" style="zoom:67%;" />

**（2）.LABEL**

<img src="https://s2.loli.net/2024/01/08/ZKR7JCEc5oVl9Xu.png" style="zoom:67%;" />

##### 5.Validation of the model on the test set

<img src="https://s2.loli.net/2024/01/08/lsOQqBxgaNXdrhV.png" style="zoom: 67%;" />

##### 6.ROC changes of six types of bacteria in the model

<img src="https://s2.loli.net/2024/01/08/XdpoQ1k7H6Vc58u.png" style="zoom:67%;" />

<img src="https://s2.loli.net/2024/01/08/5iNLowIE3prUZQv.png" style="zoom: 67%;" />

```
The ROC curve can help us understand the performance of the classifier under different thresholds, as well as the sensitivity and specificity of the classifier under different classification thresholds. The abscissa of the curve is the False Positive Rate (False Positive Rate), which is the proportion of samples that are incorrectly classified as positive to all negative samples. The ordinate of the curve is the True Positive Rate (True Positive Rate), which is the proportion of samples that are incorrectly classified as positive. The proportion of samples to all positive samples. The closer the curve is to the upper left corner, the better the performance of the classifier.
Through the ROC curve, we can judge whether the performance of the classifier is good enough. We can also compare the performance of multiple classifiers and select the best classifier.
For example, if the area under the ROC curve (AUC) is close to 1, it means that the performance of the classifier is better. If the area under the ROC curve is close to 0.5, it means that the performance of the classifier is not as good as random guessing (the AUC of random guessing is 0.5).
```

### 1.2.3Accuracy improvement compared to classic networks

![](https://s2.loli.net/2024/01/08/VZyu3ijYpUPTe5I.png)

| Method   | UNLABEL | LABEL  |
| -------- | ------- | ------ |
| SVM      | 0.6981  | 0.7020 |
| RNN      | 0.8078  | 0.8104 |
| KNN      | 0.8567  | 0.8574 |
| CNN      | 0.9097  | 0.9187 |
| MBAA-CNN | 0.9897  | 0.9909 |



![](https://s2.loli.net/2024/01/08/LwjB3JteIixdM8Y.png)

## 2. Deployment method suggestions

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

