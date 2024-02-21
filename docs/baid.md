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