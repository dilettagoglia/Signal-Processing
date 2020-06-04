# ISPR course 
(Intelligent Systems for Pattern Recognition, A.Y. 2019/20, University of Pisa)

## Midterm 1 - Assignment 5 (march 2020)
### Image processing with SIFT algorithm

- Select one image from each of the eight thematic subsets (see previous assignment), for a total of 8 images.
- Extract the SIFT descriptors for the 8 images using the visual feature detector embedded in SIFT to identify the
points of interest. 
- Show the resulting points of interest overlapped on the image. 
- Then provide a confrontation between two SIFT descriptors showing completely different information 
(e.g. a SIFT descriptor from a face portion Vs a SIFT descriptor from a tree image). 
- The confrontation can be simply visual: for instance you can plot the two SIFT descriptors closeby as barplots 
(remember that SIFTs are histograms). But you are free to pick-up any
reasonable means of confronting the descriptors (even quantitatively, if you whish).


### Useful sources
- [Histogram Comparison](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html#theory)
- [Histogram Normalization](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#normalize)
- [OpenCV](https://books.google.it/books?id=LPm3DQAAQBAJ)


### Further informations
- Project grade: from 0 to a maximum of 3 points.
- Grade obtained: 3/3




## Midterm 2 - Assignment 1 (april-may 2020)
### Hidden Markov Models for regime detection
- Fit an Hidden Markov Model with Gaussian emissions to the data in [DSET1](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#); 
it is sufficient to focus on the “Appliances” and “Lights” columns of the dataset which measure the energy consumption of appliances and lights, respectively, across a period of 4.5 months. 
- Consider the two columns in isolation, i.e. train two separate HMM, one for appliances and one for light. 
- Experiment with HMMs with a varying number of hidden states (e.g. at least 2, 3 and 4). 
- Once trained the HMMs, perform Viterbi on a reasonably sized subsequence (e.g. 1 month of data) and plot the timeseries data highlighting (e.g. with different colours) the hidden state assigned to each timepoint by the Viterbi algorithm. 
- Then, try sampling a sequence of at least 100 points from the trained HMMs and show it on a plot discussing similarities and differences w.r.t. the ground truth data.

### Useful sources
- [Fitting Hidden Markov Models](https://waterprogramming.wordpress.com/2018/07/03/fitting-hidden-markov-models-part-ii-sample-python-script/)
- [States and HMMs](https://www.analyzemydata.org/analyses/2017/10/6/states-and-hmms)
- [HMM implemented by hmmlearn](https://tomaxent.com/2017/05/01/HMM-implemented-by-hmmlearn/)
- [Building Hidden Markov Models for Sequential Data](https://www.youtube.com/watch?v=Fc-up710V9A&t=177s)
- [Introduction to Hidden Markov Models with Python Networkx and Sklearn](http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017)


### Further informations
- Project grade: from 0 to a maximum of 3 points.
- Grade obtained: 3/3


## Midterm 3 - Assignment 3 (may 2020)
### Gated RNN for timeseries prediction

[DATASET](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#) used 

Train a gated recurrent neural network (LSTM) to predict energy expenditure (“Appliances” column) using two approaches:
- Predict the current energy expenditure given as input information the temperature (T_i) and humidity (RH_i) information from all the i sensors in the house.
- Setup a one step-ahead predictor for energy expenditure, i.e. given the current energy consumption, predict its next value.

Show and compare performance of both methods.

### Useful sources
- [Tuning LSTM hyperparameters](https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)
- [Reshape data for LSTM](https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00)
- [Timeseries forecasting with LSTM](https://www.curiousily.com/posts/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python/)
- [LSTM for Timeseries](https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba)

### Further informations
