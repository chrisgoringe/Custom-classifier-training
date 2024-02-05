# Two stage training

Concept - a second model which is trained just on images that "pass" the first model might be able to learn the finer points that make a good image great.

- Stage 1 - "coarse aesthetic judgement"
    - Generated 790 images 
    - AB scored them
    - Trained primary aesthetic model on those 790 image-scores

- Stage 2 - "fine tuning top end"
    - Generated 219 "top end" images that scored at least 0.6 on the primary model
    - AB scored them
    - Trained secondary aesthetic model on those 219 image-scores

Testing: applied each model to each training set to get spearman coefficient. As might be expected, each model was very strong on the dataset it was trained on, and fairly strong on the other dataset. The first model is better on the second imageset than vica versa, which makes sense, since the second model didn't have any exposure to low or mid end images.

|image-set|model|spearman|notes|
|-|-|-|-|
|1|1|0.77|
|1|2|0.28|
|2|1|0.34|
|2|2|0.61|less strong that 1-1; we're making finer judgements?|

- Stage 3 
    - Generated 100 images and kept those which scored at least 0.6 on one of the models
    - AB scored them

|Passed primary|Passed secondary|#|Average Score|
|-|-|-|-|
|Yes| - |39| 0.034|
|Yes|No|22|-0.039|
|Yes|Yes|17| 0.128|
|-|Yes|20| 0.042|
|No|Yes|3|-0.446|

The secondary filter increased the average score from 0.034 to 0.128...


# MSE

Train one-pass on 790 images with mse. `Loss 0.2602, spearman 0.8772 (p=2.8e-253), pearson 0.8784 (p=6.7e-255)` (better than a model trained using ranking loss!)
With high-end-fix. 
`Loss 0.2534, spearman 0.8908 (p=3.8e-272), pearson 0.8923 (p=1.8e-274), AB  86.33%` -> new.safetensors
`Loss 0.2573, spearman 0.8887 (p=4.3e-269), pearson 0.8911 (p=1.3e-272), AB  86.24%` -> new2.safetensors

Compare
 primary.safetensors
`spearman 0.2777 (p= 1.9e-15), pearson 0.2924 (p= 4.8e-17), AB  59.33%` secondary.safetensors


small - `Loss 0.262, spearman 0.8343`
small high end fix - `loss 0.260, spearman 0.8743 (p=1.5e-249), pearson 0.8748 (p=  3e-250)`

## High end fix

Based on the logic above: simultaneously train two netorks, `primary` and `high_end`.

score = `primary + high_end * (1+torch.tanh(5*(primary-0.6))) * 0.5`

or, variable_hef: score = `primary + high_end * weight_fn(primary) * 0.5` with `nn.Sequential( nn.Linear(1,1), nn.Tanh(), nn.Linear(1,1) )`

(in super2, initialise the Sequential to match the non-variable version)

## Test

All trained on original 790 images, and then tested on the 100 new images

|model|loss|stats (on new dataset)|
|-|-|-|
|primary|0.2550|spearman 0.5084 (p= 6.7e-08), pearson 0.4711 (p= 7.6e-07), AB  68.14%|
|primary2|0.2597|spearman 0.5108 (p= 5.7e-08), pearson 0.4983 (p= 1.3e-07), AB  68.16%|
|primary3|0.2595|spearman 0.4790 (p= 4.6e-07), pearson 0.4585 (p= 1.6e-06), AB  67.13%|
|new|0.2534|spearman 0.5497 (p= 3.2e-09), pearson 0.5276 (p= 1.7e-08), AB  69.90%|
|new2|0.2572|spearman 0.5157 (p=   4e-08), pearson 0.4964 (p= 1.5e-07), AB  68.59%|
|new3|0.2588|spearman 0.5291 (p= 1.5e-08), pearson 0.5081 (p= 6.8e-08), AB  69.19%|
|super|0.2605|spearman 0.5180 (p= 3.4e-08), pearson 0.4953 (p= 1.6e-07), AB  68.63%|
|super2|0.2564|spearman 0.4874 (p= 2.7e-07), pearson 0.4843 (p= 3.3e-07), AB  67.49%|
|super3|0.2575|spearman 0.5158 (p=   4e-08), pearson 0.4963 (p= 1.5e-07), AB  68.53%|