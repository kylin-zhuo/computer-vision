#! /bin/bash

echo $0
echo $1
echo $2

# parameters:
# 1 - the positive image
# 2 - background file
# 3 - number of positive images to create
# 4 - window width
# 5 - window height
# 6 - number of positive images used for training
# 7 - number of negative images used for training 
# 8 - number of stages

opencv_createsamples -img $1 -bg $2 -info pos/info.lst -pngoutput pos/ -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num $3 
opencv_createsamples -info pos/info.lst -num $3 -w $4 -h $5 -vec positives.vec
opencv_traincascade -data data -vec positives.vec -bg $2 -numPos $6 -numNeg $7 -numStages $8 -w $4 -h $5
