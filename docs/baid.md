# Baid

Series of training runs.

Loss model mse; metric for scoring spearman, set for scoring eval.

```
--trials            =100
train_epochs        : (4, 10)
warmup_ratio        : (0.06, 0.24)
log_lr              : (-4.4, -3.0)
batch_size          : (50, 200)
first_layer_size    : (600, 1300)
second_layer_size   : (300, 1500)
dropout             : (0.0, 0.2)
input_dropout       : (0.0, 0.06)
output_dropout      : 0.0
```

|layers|best trial|median trial|
|-|-|-|
|0|0.2107|0.2041|
|1|0.2200|0.2127|
|2|0.2204|0.2119|
|3|0.2169|0.2089|
|4|||
|5|||
|6|||
|7|||
|8|||