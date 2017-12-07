### Step 1: Collection and preprocessing of Images

Execute the function `utils.py` for collecting and preprocessing the image data.

#### A) to download the images from `image_link` to `directory`

`python utils.py download directory image_link`

for example,

`python utils.py download neg/ http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02726681`

The program uses 100 * 100 gray scale images as samples.

#### B) to clean the downloaded images contained in `directory` according to the patterns in folder `noises_directory`

`python utils.py clean directory noises_directory`

The `noises` folder need to be maintained by manually adding the noised images into it.

#### C) to create background file for the images used as the negative samples in `directory`

`python utils.py create_background directory`

### Step 2: Generating Positive Images and Training Cascade classifiers

Run the file `run.sh` which contains the batch commands to start the process of composing the positive images, generating the vector files and training the Haar classifiers.

The parameters to pass in to `run.sh` are:

```
$1 - the positive image
$2 - background file
$3 - number of positive images to create
$4 - window width
$5 - window height
$6 - number of positive images used for training
$7 - number of negative images used for training
$8 - number of stages
```
For example:

`./run.sh train/hand.jpg bg.txt 2000 20 20 1800 900 10`

After the training, the cascade file can be obtained from `data/cascade.xml`. Manually modified the name of the classifier and add it to the folder `cascades/` for future usage.

### Step 3: Running the main function for object detection

The range of objects to detect depends on the availability of the cascade files in `cascades/`. Run `python main.py` and follow the instruction to test the program.
