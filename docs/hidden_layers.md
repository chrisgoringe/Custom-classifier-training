# Different hidden layers

Training on training4 with 

```python
    "meta_trials"       : 200,
    "sampler"           : "CmaEs", 
    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),
    "half_batch_size"    : (1, 50),           
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), ],
    "hidden_layers"      : [ (10, 1000), (10, 1000), ],
```

Hidden layer 0 is the default behaviour

|Hidden layer(s)|Loss|AB|Spearman|
|-|-|-|-|
|0|0.2721|77.96%|0.7631|
|0+0|0.2723|77.71%|0.7598|
|1|0.2714|77.84%|0.7631|
|0+1|0.2678|78.09%|0.7694|
|2|0.2762|77.99%|0.7626|
|0+2|0.2704|77.83%|0.7627|
|0+1+2|0.2702|77.95%|0.7636|
|3|0.2857|77.07%|0.7416|
|16|0.3558|73.00%|0.6577|

Trained weights of last n layers - 500 trials

|n|Loss|AB|Spearman|
|-|-|-|-|
|2||||
|3|0.2625|78.11%|0.7665|
|4||||
