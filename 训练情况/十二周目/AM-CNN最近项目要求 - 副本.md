# AM-CNN最近项目要求

### 1.更换对比算法为常用算法（SVM,RNN,CNN,KNN）利用Label数据训练

（1）更换算法测试结果

**AM-CNN:**

<img src=".\AM-CNN.png" alt="AM-CNN" style="zoom:50%;" />

**SVM:**

<img src=".\SVM.png" alt="SVM" style="zoom:52%;" />



**RNN:**

<img src=".\RNN.png" alt="RNN" style="zoom:50%;" />



**KNN:**

<img src=".\KNN.png" alt="KNN" style="zoom:50%;" />





**CNN:**

<img src=".\CNN.png" alt="CNN" style="zoom:50%;" />

**综合汇总图片：**

<img src=".\image-20230909163438257.png" alt="KNN" style="zoom:50%;" />
（2）评估曲线


<img src=".\image-20230909131424820.png" alt="KNN" style="zoom:50%;" />
**AM-CNN**在准确率-召回率曲线中表现**稳定**；

同时在**N**轮训练下来

排名最高的样本数中准确率和召回率表现最好的仍然是**AM-CNN**模型



### 2.美化预测结构图（分为Label与UnLabel）


<img src=".\image-20230909091026674.png" alt="KNN" style="zoom:50%;" />


做出5种细菌的测试示例：



<img src=".\image-20230909163659828.png" alt="image-20230909163659828" style="zoom:50%;" />

### 3.将SVM,RNN,CNN,KNN以及本论文中的AM-CNN训练结果数据以表格的形式展示出来

<img src=".\image-20230909091313876.png" alt="image-20230909091313876" style="zoom:60%;" />



<img src=".\Train_Data.png" alt="Train_Data" style="zoom:50%;" />

### 4.（难）根据训练数据选择出最能代表每种细菌特征的光谱区间



待续
