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

### Short description of the code
I used SIFT **detector** to find some points of interest and then SIFT **descriptor** to compute a descriptor for each point.

The function `drawKeyPoints` draws small circles around the points overlapped on the corresponding image, and this **flag** will also show the **size** of the keypoint and the the **orientation.**

Then I selected some random descriptors to perform a comparison between completely different images.

The first kind of comparison is simply **visual**, while the second comparison instead will be **quantitative**.

The result of the first part of the analysis is a bag of SIFT descriptors for each image, so we have one descriptor (that is one histogram) for each keypoint.

For each image, the SIFT detector has found different number of points that are considered positions where an interesting feature has been detected.

Result of **visual comparison**: four descriptors are selected randomly from four completely different images and they have been compared vertically into barplots.
On the x axes of the barplot there are frequencies, while on the y axes there is how many times each frequency has been touched.
The **spectral domain** of the signal shows that lower frequencies are the most represented in the signal, and the interesting thing is that this happen for each descriptor chosen randomly.
However we lose the information about which gradient had that specific frequency.

In the second comparison, that is **quantitative**, I’ve used the `compareHist` function to get numeric parameters that express how well two histograms match with each other.
In particular, I’ve done this with three different metric, that are **correlation, chi-square and intersection**.
I have performed the comparison with three descriptors chosen randomly from different images.

First I’ve compared a descriptor with itself to test the accuracy of the methods, and then I’ve compared the same with two different ones.
The interesting result is that these numbers proves that descriptors are actually different, but they seem to be representative of the overall image: a horse seems to be more similar to a cow than a car, and this happens for most of the descriptors chosen randomly.

The weak aspect of this analysis is… How reliable is this evidence? Can we say that an histogram randomly selected into a bag of descriptors is really a representative sample of the whole image?

## Useful sources
- [Histogram Comparison](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html#theory)
- [Histogram Normalization](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#normalize)
- [OpenCV](https://books.google.it/books?id=LPm3DQAAQBAJ)


## Further informations
- Project grade: from 0 to a maximum of 3 points.
- Grade obtained: 3/3
