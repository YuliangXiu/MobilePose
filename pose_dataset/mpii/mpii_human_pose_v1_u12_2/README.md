--------------------------------------------------------------------------- 
MPII Human Pose Dataset, Version 1.0 
Copyright 2015 Max Planck Institute for Informatics 
Licensed under the Simplified BSD License [see bsd.txt] 
--------------------------------------------------------------------------- 

We are making the annotations and the corresponding code freely available for research 
purposes. If you would like to use the dataset for any other purposes please contact 
the authors. 

### Introduction
MPII Human Pose dataset is a state of the art benchmark for evaluation
of articulated human pose estimation. The dataset includes around
**25K images** containing over **40K people** with annotated body
joints. The images were systematically collected using an established
taxonomy of every day human activities. Overall the dataset covers
**410 human activities** and each image assigned an activity
label. Each image was extracted from a YouTube video and provided with
preceding and following un-annotated frames. In addition, for the test
set we obtained richer annotations including body part occlusions and
3D torso and head orientations.

Following the best practices for the performance evaluation benchmarks
in the literature we withhold the test annotations to prevent
overfitting and tuning on the test set. We are working on an automatic
evaluation server and performance analysis tools based on rich test
set annotations.

### Citing the dataset
```
@inproceedings{andriluka14cvpr,
               author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
               title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
               booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
               year = {2014},
               month = {June}
}
```

### Download

-. **Images (12.9 GB)**
   
   http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
-. **Annotations (12.5 MB)**	
   
   http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12.tar.gz
-. **Videos for each image (25 batches x 17 GB)**	

   http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_batch1.tar.gz
   ...
   http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_batch25.tar.gz
-. **Image - video mapping (239 KB)**	
   
   http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_keyframes.mat

### Annotation description 
Annotations are stored in a matlab structure `RELEASE` having following fields

- `.annolist(imgidx)` - annotations for image `imgidx`
  - `.image.name` - image filename
  - `.annorect(ridx)` - body annotations for a person `ridx`
		  - `.x1, .y1, .x2, .y2` - coordinates of the head rectangle
		  - `.scale` - person scale w.r.t. 200 px height
		  - `.objpos` - rough human position in the image
		  - `.annopoints.point` - person-centric body joint annotations
		    - `.x, .y` - coordinates of a joint
		    - `id` - joint id 
[//]: # "(0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)"
		    - `is_visible` - joint visibility
  - `.vidx` - video index in `video_list`
  - `.frame_sec` - image position in video, in seconds
 
- `img_train(imgidx)` - training/testing image assignment 
- `single_person(imgidx)` - contains rectangle id `ridx` of *sufficiently separated* individuals
- `act(imgidx)` - activity/category label for image `imgidx`
  - `act_name` - activity name
  - `cat_name` - category name
  - `act_id` - activity id
- `video_list(videoidx)` - specifies video id as is provided by YouTube. To watch video on youtube go to https://www.youtube.com/watch?v=video_list(videoidx) 

### Browsing the dataset
- Please use our online tool for browsing the data
http://human-pose.mpi-inf.mpg.de/#dataset
- Red rectangles mark testing images

### References
- **2D Human Pose Estimation: New Benchmark and State of the Art Analysis.**

  Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler and Bernt Schiele. 

  IEEE CVPR'14
- **Fine-grained Activity Recognition with Holistic and Pose based Features.**

  Leonid Pishchulin, Mykhaylo Andriluka and Bernt Schiele.

  GCPR'14

### Contact
You can reach us via `<lastname>@mpi-inf.mpg.de`
We are looking forward to your feedback. If you have any questions related to the dataset please let us know.
