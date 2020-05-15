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

