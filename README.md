# vehicle_detection_hog_svm


### ◆ Histogram of Oriented Gradients (HOG)
#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cell In [280] of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters while training my classification model, and empirically found that the parameters mentioned aboce resulted in the highest accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all available color-space channels (0,1 and 2) along with the HOG features, spatial features and color histograms.the accuracy of the classifier reached 0.989%. Also I used StandardScaler for feature normalization along resulting dataset. I found that it is really importaint step.
(The training process can be seen in the cell In[557] in P5_vehicle_detection.ipynb. )


### ◆ Sliding Window Search
#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have implemented sliding window search in 573 and 574 cells of IPython Notebook (functions slide_window and search_windows).I used windows with sizes 64x64. Also, It was difficult to implement filter for false positive detections.My solution was to increce the scale from 1 to 1.5. This was good solution.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

After searching on all the scales indicated above, using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, the results are satisfying. The test images can be shown below:

**Test1,2,3**

![alt text][image4]

**Test4,5,6**

![alt text][image5]



---

### ◆ Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I used`scipy.ndimage.measurements.label()` function to identify individual blobs in the heatmap. The implementation could be found on In [576]


### ◆ Discussion
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Fine-tuning HOG features extraction and sliding windows parameters was one of the most time-consuming task. In order to properly adjust parameters I have conducted a number of experiments to find out what combination of parameters works best. It was difficult to implement filter for false positive detections.My solution was to increce the scale from 1 to 1.5. This was good solution.

I'm going to try to detect the cars using convolutional neural network.



[//]: # (Image References)
[image1]: ./project_images/car_notcar.png
[image2]: ./project_images/hog.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./project_images/test123_output.png
[image5]: ./project_images/test456_output.png
[video1]: ./project_output.mp4

-----------
