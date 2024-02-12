# Training 1 and 4

- Set 1, 790 images
- Set 4, 1000 images

both sets fully AB trained

model trained on each set

||model1|model4|
|-|-|-|
|set1|73.19%|70.05%|
|set4|72.50%|77.51%|

## details

test images from training1
model1
                     :   201 images, db score  0.057 +/- 0.67, model score  0.013 +/- 0.45, spearman 0.6499 (p= 1.6e-25), pearson 0.6540 (p= 6.5e-26), AB  73.19%
model4
                     :   201 images, db score  0.057 +/- 0.67, model score  2.498 +/- 1.18, spearman 0.5752 (p= 4.2e-19), pearson 0.5746 (p= 4.7e-19), AB  70.05%
                                          
test images from training4
model1
                     :   238 images, db score  0.085 +/- 0.78, model score  0.008 +/- 0.39, spearman 0.6322 (p= 5.8e-28), pearson 0.6203 (p= 1.1e-26), AB  72.50%
model4
                     :   238 images, db score  0.085 +/- 0.78, model score  2.141 +/- 1.22, spearman 0.7560 (p= 2.5e-45), pearson 0.7357 (p= 8.2e-42), AB  77.51%



##




    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),
    "half_batch_size"    : (1, 50),   
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), ],
    "hidden_layers"      : [ (10, 1000), (10, 1000), ],
    mse=0.272, ab=77.96% (test split only
    
    
    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),
    "half_batch_size"    : (1, 50),   
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), (0.0, 0.8), (0.0, 0.8),],
    "hidden_layers"      : [ (10, 100),  (10, 100),  (10, 100),  (10, 100), ],)
    0.278, 77.34


    "num_train_epochs"   : (15, 30),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-4.5, -3),
    "half_batch_size"    : (20, 50),            
    "dropouts"           : [ (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),],
    "hidden_layers"      : [ (10, 100),  (10, 100),  (10, 100),  (10, 100), ],
    0.279


    "num_train_epochs"   : (15, 30),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-4.5, -3),
    "half_batch_size"    : (20, 50),            
    "dropouts"           : [ (0.0, 0.5), (0.0, 0.5), ],
    "hidden_layers"      : [ (1000, 2000), ],
    0.276, 78.15


    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), (0.0, 0.8) ],
    "hidden_layers"      : [ (10, 1000), (10, 1000), ],