# ART_Line_Detection
Line detection for the autonomous rover 2018

We perform lane detection on the imput images via an image processing pipeline that receives as input:
![alt inputimage](https://github.com/UTRA-ART/ART_Line_Detection/blob/master/images/3.jpg)

Warps to birds-eye view, generating a thresholded binary image
![alt outputimage](https://github.com/UTRA-ART/ART_Line_Detection/blob/master/output_images/3.jpg)

Then using a sliding-window algorithm which polyfits in horizontal and vertical directions:
![alt finalimage](https://github.com/UTRA-ART/ART_Line_Detection/blob/master/final_images/3.png)
