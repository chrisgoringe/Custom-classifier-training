# Model shortcut

Using a model to shortcut AB training.

## Background

- I took a set of 400 images, trained to have 2000 comparisons
- I trained a model on them, and looked at the spotlight output (spearman for training was 0.7087)
- It seemed as if, for the outliers at least, the model ranking was better than the DB
- So I did one more AB step: Average p value for chosen result 73.7315%, spearman start-end: 0.9406
- Then I did an AB step starting from the model scores: Average p value for chosen result 77.6218%, spearman start-end: 0.9526

The model scores seem to be better converged! How far back does this go?

- trained a model on the scores after 1000 comparisons (spearman for training was 0.5861 - the model was much less confident...)
- AB step from model scores: spearman start-end: 0.8452  (AB training had 0.9363 at this point)