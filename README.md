# WinkDetect
Wink detection using Harr Like cascade

## Description
* This contains 2 files which are almost same but due to the nearest neighbours the perm differently.
* To detect wink we have considered the presence of only one eye in the face.
* Steps Taken
  1. Get all possible faces. Here we allow all false positives faces.
  2. Compute the Region for possible Eye region.
  3. For each region we check for presence for Eye. If there contains one we capture it. Here we are more conserned about the ratio of False positive and True Negatives.
  4. If only one of the two region contains eye than we say it was a wink. 

## Bad Image
1. If the Image has low brightness or too high brightness than the detection may not take place.
2. If the Image has non smooth pixels, i.e. to sharp edges than false positives increases and count of eyes go up.


## Enviornment
1. python 3.7
2. opencv-python==4.0.0.21
3. numpy==1.16.1


## Execution Instruction
>To run on folder
```
python Filename.py folder/that/contains/images/
```
>To run on camera
```
python Filename.py
```



