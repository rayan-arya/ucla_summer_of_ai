import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])  #random 50-50 coin toss

fair_probs = torch.tensor([0.5, 0.5]) 
print(Multinomial(100, fair_probs).sample()) #100 simulations with exactly 50-50 odds

print(Multinomial(100, fair_probs).sample() / 100) #frequency of the 0s and 1s

counts = Multinomial(10000, fair_probs).sample()
print(counts / 10000) #same thing with more draws

counts = Multinomial(1, fair_probs).sample((10000,))
cumulative_counts = counts.cumsum(dim=0)
estimates = cumulative_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated Probability')
d2l.plt.legend(); 
d2l.plt.show() #graph probability of the fair_probs method for 10,000 attempts
