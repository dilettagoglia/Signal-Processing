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
