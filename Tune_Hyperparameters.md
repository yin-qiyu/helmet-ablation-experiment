🧹🧹🧹

不推荐使用梯度下降来设置学习率等超参数，成本高

The first thing we need to define is the `method` for choosing new parameter values.

We provide the following search `methods`:

- **`grid` Search** – Iterate over every combination of hyperparameter values. Very effective, but can be computationally costly.
  - 网格搜索：迭代超参数值的每个组合。非常有效，但计算成本可能很高。

- **`random` Search** – Select each new combination at random according to provided `distribution`s. Surprisingly effective!
  - 根据提供的“分布”随机选择每个新组合。出乎意料的有效！
- **`bayes`ian Search** – Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Works well for small numbers of continuous parameters but scales poorly.
  - 根据超参数创建度量分数的概率模型，并选择具有提高度量的高概率的参数。适用于少量连续参数，但扩展性很差。

We'll stick with `random`.