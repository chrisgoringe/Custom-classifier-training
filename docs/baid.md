# Baid

Series of training runs.

Loss model mse; metric for scoring spearman, set for scoring eval.

```
--trials            = 100
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

## ChrisGoringe/vitH16

|layers|best trial|median trial|
|-|-|-|
|0|0.2107|0.2041|
|1|0.2200|0.2127|
|2|0.2204|0.2119|
|3|0.2169|0.2089|
|4|0.2150|0.2088|
|5|0.2139|0.2075|
|6|0.2112|0.2048|
|7|0.2094|0.2026|
|8|0.2071|0.2015|

## ChrisGoringe/bigG-vision-fp16

|layers|best trial|median trial||
|-|-|-|-|
|0|0.2226|0.2135||
|1|0.2286|0.2181||
|2|0.2260|0.2190||
|3|0.2255|0.2186||
|4|0.2283|0.2198||
|5|0.2221|0.2155||
|6|0.2203|0.2135||
|7|0.2171|0.2113||
|8|0.2196|0.2120||
|join(1,2,3,4)|0.2292|0.2238||
|weight(1,2,3,4)|0.2136|0.1929|0.2413, -0.0704,  0.3854,  0.0740, -0.2325|
|fixed_weight(1,2,3,4)|0.2210|0.2099|0.2413, -0.0704,  0.3854,  0.0740, -0.2325|
|weight(1,3,5,7)|0.2173|0.1900|0.43631, 0.09442, 0.36981, 0.06868, 0.24160|
|fixed_weight(1,3,5,7)|0.2190|0.2117|0.439, 0.096, 0.370, 0.071, 0.242|
|weight(1,5)|0.2087|0.1922|0.54672,  0.10667, -0.32514|


## ChrisGoringe/aim-600M-fp16

Default behaviour average(9,8,7,6,5,4)

|layers|best trial|median trial||
|-|-|-|-|
|average(9,8,7,6,5,4)|0.1884|0.1825||
|average(0,1,2,3,4,5)|0.1884|0.1807||
|average(10,11,12,13,14,15)|0.1831|0.1782||
|weight(9,8,7,6,5,4)|0.1798|0.1388|-6.0974, -1.2978, -5.2508, -0.9428, -3.4215, 1.0000|
|join(9,8,7,6,5,4)|0.1889|0.1836||

## ChrisGoringe/aim-3B-fp16

|layers|best trial|median trial||
|-|-|-|-|
|average(9,8,7,6,5,4)|0.1791|0.1743||