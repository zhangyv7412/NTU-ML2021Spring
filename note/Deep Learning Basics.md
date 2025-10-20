#Deep-Learning   

# 0. Object  
在深度学习中, 我们实际上是在拟合函数.  
为了拟合出足够复杂的函数, 我们使用大量非线性函数去拟合, 添加大量参数来提供足够多的自由度让拟合出的函数充分符合我们的要求.  

# 1. Types of ML  
1. Regression  
2. Classification  
3. Structured Learning

# 2. Tayler Series Approximation
$L(\theta)$ around $\theta = \theta^*$ can be approximated as:
$$L(\theta) \approx L(\theta^*) + (\theta - \theta^*)^T g + \frac{1}{2}(\theta - \theta^*)^T H (\theta - \theta^*)$$
$$g = \nabla L(\theta^*)$$
$H$ is Hessian-Matrix:
$$H_{ij} = \frac{\partial^2}{\partial \theta_i \partial \theta_j} L(\theta^*)$$

# 3. Adaptive Learning Rate  
## 3.1 Root Mean Square  
目的是在坡度小的地方加快训练, 坡度大的地方减慢训练防止震荡.  
基于上述目的, 我们希望学习率和梯度成反比.  
$$ l = \frac{\eta}{\sigma_i^t}$$
$$\sigma_i^t = \sqrt{\frac{1}{n + 1} \sum^{t}_{i = 0}(g_i^t)^2}$$  
RMS函数的缺点在于, 对于单一参数, 只能适应一种风格的"地貌", 如果一会坡度大一会坡度小, 它就会没那么好用.  
为了解决这个问题, 人们提出了RMSProp.  
## 3.2 RMSProp  
在RMS中, 我们给每一步算出的梯度相同的权重, 在RMSProp中则把权重参数化, 我们可以通过调节这个Hyperparaemter来决定最新的梯度和之前的梯度的权重分别是多少.  

这么做的好处在于, 不同参数的学习率可以在面对不同的"地貌"时迅速调整到合适的大小.    
$$ l = \frac{\eta}{\sigma_i^t}$$
$$\sigma_i^0 = \sqrt{(g_i^0)^2}$$
$$\sigma_i^1 = \sqrt{\alpha (\sigma_i^0)^2 + (1 - \alpha)(g_i^1)^2}$$
$$\vdots$$
$$\sigma_i^t = \sqrt{\alpha (\sigma_i^{t - 1})^2 + (1 - \alpha)(g_i^t)^2}$$

## 3.3 Learning Rate Scheduling  
### 3.3.1 Learning Rate Decay
$$ l = \frac{\eta}{\sigma_i^t}$$
$\eta$替换为$\eta(t)$, 其函数图像是从比较高的位置开始, 随时间越来越小.  
这么设计的想法是, 随着时间推移我们应该距离最优点越来越近, 所以把步长变短, 变化更精确.
### 3.3.2 Warm Up
$\eta$替换为$\eta(t)$, 其函数图像是从较小数值开始, 随时间先变大后变小.  
因为有效所以在很多地方被使用, 至于其作用原理目前仍然是开放性问题.  
一个可能的解释是, 一开始让学习率较小使得不偏离原点太远, 在统计足够多数据之后$\sigma_i^t$会更加精准.

# 4. Optimizer  
## 4.1 Adam  
Adam = RMSProp + Momentum  

# 5. Cross Entropy  
##### Minimizing Cross Entropy is equivalent to maximizing likelihood.  
在Classification任务中, Cross Entropy更常用作Loss.  


