# mu sigma

Convert the model to have two 'lobes', generating mu and sigma. The model is predicting a probability distribution centered on mu, with standard deviation sigma.

We maximise the joint probability of all the results by minimising the negative log likelihood:

```python
    loss_fn = torch.nn.GaussianNLLLoss()
    loss = loss_fn(prediction,actual,torch.square(sigma))
```
