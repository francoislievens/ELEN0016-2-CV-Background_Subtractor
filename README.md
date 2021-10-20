# ELEN0016-2-CV-Background_Subtractor

### Julien Hubar, Matthias Pirlet, François Lievens

This file contain the implementation of our
background subtractor project for the ELEN0016 lecture.

For this project, we are performing the following operations on each video frame:

- We are using our GaussianSubtractor class (file GaussianSubtractor.py) to detect pixels
who significantly change of value between frames. At the begin of the execution, we 
  are instancing an GaussianSubtractor object and we push in it a buffer who contain
  the 100 first frames of the video.
  At each iteration, this object will compute the mean and the standard deviation
  of all pixels over the 100 frames in the buffer. If a pixel of the reading frame
  is far from the buffer mean than 2.5 times the standard deviation, we consider that it's a part
  of a moving object. After that, the reading frame is push himself in the buffer.
  
- Since this is done, we are performing a delation operation over obtained mask.
This operation permit us to obtain a closed circule mask for each droplet, who
  mandatory to perform the next operation. The delation process that we perform use
  a 3x3 squared kernel. The implementation of this will be discus in next parts.
  
- Since we obtain a closed circle for each droplet, we are using the floodFill operation
of the library OpenCV. This will fill the mask to cover the center of each
  droplet. We have not reimplement this operation. 
  
- We obtain now a mask who contain the full droplet, but also a lot of noise.
To avoid this noise, we have reimplement the opening who consist of erosion
  followed by a delation. Since we already done delation with a kernel of 3x3 px,
  we are performing an erosion with a kernel of 6x6px, followed by a delation of
  an only 3x3 kernel in order to recover the original droplet size.
  
- The erosion and delation operations are done by using our MorphoOperator class.
In our implementation, we have chosen to use a squared kernel, who already give us
  good results, but we can imagine that results can be improved by using circular kernels.
  


An improvement that we can imagine, for this type of video who always share
exactly the same background can be to not update the buffer at each frame.
If we proceed like that, we have not to compute again the mean and the standard
deviation for each pixels of images of the buffer.

<p align="center">
  <img src= https://github.com/francoislievens/ELEN0016-2-CV-Background_Subtractor/blob/main/output_frames/group2_00022.jpg/>
</p>
