=======================================================================
Computer Vision Image Blending Experiments with Pyramids
=======================================================================

In this repo I am implementing Image pyramids based on the References bibliography 
and then I am doing some Image blending Experiments with the pyramids method.



Description of the proposed method for blending two images
============

Given two images and a suitable mask we can perform a blending
of the images as follows: I1(n)M(n) + I2(n)(1−M(n)). However this
trivial way shows the artifact of the abrupt change at the point
where we blend the two images. So what we need to do is smooth out
the sharp transition of our mask but without losing the details
of our final image. The method we use in this repository to acomplish this
is described bellow:

#. We compute the L+1 level Gaussian pyramid of our mask. With this one
   way at each level we have a smoother transition.

#. We calculate the Laplacian pyramids of L+1 levels of the images we want to
   blend. In this way we have the details (high frequencies) of the images.

#. Now we make a pyramid which in every scale up to level L contains
   the mixing (which is done with the smooth masks) of the high
   image frequencies and in L+1 contains the mixing of the low frequencies of
   images, then we can start from L+1 scale and go to our original scale (level
   0 ) and at each level to add the finer details.


The transition from one level of high resolution to the next low resolution is based on the fact that since
we have filtered our image with a low pass filter then the information of the lows
frequencies is lost and therefore if we subsample we will get the original image
just on a higher scale (e.g. if the original image was in cm^2 the resulting image will be
in m^2 zoom-out). The same effect can be achieved with geometric 
scaling transformations except that in this case we do not reduce the number of
of our samples on the grid.


.. Image:: /Documentation_Images/Pyramid_funcs.png





orange_apple_experiment
============


**ORANGE PYRAMID:**

.. Image:: /Documentation_Images/Orange_Apple/Orange_Pyr.png


**APPLE PYRAMID:**

.. Image:: /Documentation_Images/Orange_Apple/Apple_Pyr.png


**BLENDED PYRAMID:**

.. Image:: /Documentation_Images/Orange_Apple/B_Pyr.png

         
The success of the method lies in having the desired image
(the blend of the two images with smooth transition) in high
scale at level 4 and we also have fine details or high frequencies
(of the superposition of the images with smooth transition) in the
levels (smaller scales) 0 , 1 , 2 , 3 .     


woman_hand_experiment
============

The parameters used are:

* Maximum pyramid scale level (Pyr_lvl) = 2

* Filter for building the pyramids of the images: 
  filter length = 5x5, Variance=1

* Filter for making the mask pyramid: 
  filter length = 20x20, Variance=10

**Woman_Hand blending result:**

.. Image:: /Documentation_Images/Woman_Hand/Woman_Hand_result.png

The mask is found after first finding the exact location (square)
which contains the eye in the woman image. What mattered a lot was
the choice of build filter of the pyramid of the mask. It is obvious
that as you increase the Variance of the Gaussian kernel the more
we will smooth out the sharp transition from one image to another.
However, if we increase the Variance too much then we will inevitably
spoil the eye as well (not just the sudden change).



custom_blending_experiment
============

The process I followed to get a custom blending of my images is as follows:

#.  I found the coordinates of the objects that I wanted my final
    image to consist of (e.g. cat, dog, bench) and then I cropped
    these objects.

#.  For each object I perform appropriate geometric transformations
    (translation and scaling) to place it in some desired position
    in my final image and then I find the masks for each object.

#.  Each image I found in the previous step contains the object in a
    square and all other pixels are 0. I replaced all those zeros
    with the mean value over all the images that was included in the
    blending. (I did this step because in the Laplacians of object
    images showed this big change from 0 (outline) of cropped
    object to some value of its pixels (square containing the object)) .


Finally after I have done all the previous steps (calculation of object
images and masks), I perform the blending sequentially. More specifically,
I choose every time my background to be the result of a previous blending
(where in the first blending I put the P200 as background and the bench
as foreground) and follow the same procedure as the one in the first blendings (orange-apple, woman-hand).


And the following result is produced for parameters :

* Maximum pyramid scale level (Pyr_lvl) = 7

* Filter for building the pyramids of the images: 
  filter length = 5x5, Variance=1

* Filter for making the mask pyramid: 
  filter length = 31x31, Variance=30              

**Custom_Blending result:**

.. Image:: /Documentation_Images/Custom_Blending/Custom_Blending_res.png



Reproduce the Experiments
============

Firstly we should mention that you have to instal all the requiered python libraries and 
to load correctly the images that provided in this repo in order for the code to work.
To reproduce the experiments you have to execute the following commands in each case: 

* orange_apple_experiment ::

   python orange_apple_experiment.py


* woman_hand_experiment ::

   python woman_hand_experiment.py


* custom_blending_experiment ::
   
   python custom_blending_experiment.py


References
====================

#. J.M. Ogden, E.H. Adelson, J.R. Bergen, P.J. Burt: Pyramid-based computer gra-
   phics, RCA Engineer, vol. 30(5), pp. 4-15 (1985).

#. Peter J. Burt, Edward H. Adelson: The Laplacian Pyramid as a Compact Image
   Code, IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. COM-31, NO. 4, APRIL
   1983.

#. J.M. Ogden, E.H. Adelson, J.R. Bergen, P.J. Burt: Pyramid-based computer gra-
   phics, Journal ACM Transactions on Graphics Volume 2 Issue 4, October 1983 Pages
   217-236.

Free software: MIT license
============
