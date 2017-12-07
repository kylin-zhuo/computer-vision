import urllib, urllib2
import cv2
import numpy as np
import os
import sys


def store_raw_images(directory, img_link, start=1):

    img_urls = urllib2.urlopen(img_link).read()
    pic_num = start
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for i in img_urls.split('\n'):
        try:
            urllib.urlretrieve(i, directory.rstrip('/') + "/"+str(pic_num) + ".jpg")
            img = cv2.imread(directory.rstrip('/') + "/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)

            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(directory.rstrip('/') + "/" + str(pic_num) + ".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            pass

    print("Saved " + str(pic_num - start) + " images to folder " + directory)


def clean_noise(directory, noises_directory):

    for img in os.listdir(directory):
        for ugly in os.listdir(noises_directory):
            try:
                current_image_path = str(directory).rstrip('/') + '/' + str(img)
                
                ugly = cv2.imread(noises_directory.rstrip('/') + '/' + str(ugly))
                question = cv2.imread(current_image_path)
                
                if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                
                    print('delete the noise images:')
                    print(current_image_path)
                    os.remove(current_image_path)
            
            except Exception as e:
                pass


def create_pos_n_neg(directory, dest):

    f = open(dest, 'w')

    for img in os.listdir(directory):
        line = directory + '/' + img + '\n'
        f.write(line)

    f.close()
    print("Saved the background information to " + dest)


# neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'   
# neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09618957'
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09610255'
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03100490' # transport
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09359803' # scenery
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n11553240'
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03529860' # home
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03798982' # cinema
# neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02726681'


if __name__ == '__main__':

    option = sys.argv[1]

    if option == 'download':

        # download the images
        directory = sys.argv[2]
        img_link = sys.argv[3]
        start = int(sys.argv[4])

        store_raw_images(directory, img_link, start)

    elif option == 'clean':
        directory = sys.argv[2]
        noises_directory = sys.argv[3]
        clean_noise(directory, noises_directory)

    elif option == 'create_background':

        directory = sys.argv[2]
        dest = sys.argv[3]
        create_pos_n_neg(directory, dest)

