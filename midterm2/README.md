## Short description of the slides

### SLIDE 2: PRIOR ANALYSIS
I have done a prior analysis on data, just to find some useful and interesting informations.

I plotted the mean value of consumptions for monthly and daily periods, to observe their behaviour, and I've found what I expected, for ex. that there are some regularities and ripetitive patterns over time.

Some parts of this prior analysis will turnout particularly useful when I will show conclusions, in particular the density plot for the probability distribution of our observations.

### SLIDE 3: HMM
I fitted two models to the data and I experimented in training them with different hidden states, interpreting them has hidden regimes of consumptions, as I showed the table.

Then I extracted some model parameters, like the log likelihood and the transition matrix to see their behaviour. 

I have done this for every model with all the different number of states, but in the slides I reported two examples: the transition matrix in the case of two components, represented as a Markov Chain, tells us that the probability to switch from a low regime of consumption from an high one is 97% and 94% vice versa.

More interesting is the behaviour of the log likelihood: in this example we see that it increases and so it improves with more hidden states in the model.

### SLIDE 4: VITERBI
Then I selected a subsequence of one month of data on which I performed Viterbi algorithm to predict the optimal hidden states assignment for the observations; I plotted it with colors representing the different regimes of consumptions assigned. 

In the example of two hidden states we see that medium variances of consumption are not well represented because the states are too few, so Viterbi shift immediately from low to high. And it is clear that the first values assigned as high are exactly the highest range of values in the plots of mean consumptions (showed in the prior analysis).

### SLIDE 5: VITERBI (II)
Viterbi with 7 hidden states and the subsequence plotted in the background. 

For each state, the criteria for wich Viterbi assign a different regime of consumption depends on the average value of this consumption. 

So to understand what hidden state (1st, 2nd, 3rd) is assigned to what cluster, I computed the mean value corresponding to each state and ordered the couples state-mean by mean value.

So for example in this way I understanded that the last state for appliances, that is the seventh, corresponds to the medium-high regime of consumption.

### SLIDE 6: SAMPLING

I performed the sampling of a sequence (I choose 143 point that is one full day of measurements). 

On the left side there are the new samples generated with the respective hidden states, and it is clear that with 2 states high variances of consumptions are not well represented while with 5 hidden states the situation is yet better.

On the right side there is the comparison. So to perform comparison between the new generated samples and the real data I choose to show their respective probability distributions, that is the density plot that I've shown in my prior analysis.

We can clearly see that the probability distribution of the new samples generated, that is its behaviour, is representative of the behaviour of the real observations.
And in particular, samples distributions get closer and closer to real distributions when hidden states in the model increases, but till the 5-th, that is the best representative number of hidden state in the model for this data.

In fact I noticed that with 7 hidden states, density plots of the new samples departed a little bit too much from the real data distribution.

This is a powerful and clear way to estimate the effectiveness of HMMs as generative models.
