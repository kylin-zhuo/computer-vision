<center>Fall 2017 CS512 Computer Vision project proposal</center>

# Implementation of Deep Neural Network to Describe the Image Contents
<br>



<br><br><br>

## Research Paper

Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge
IEEE Transactions on Pattern Analysis and Machine Intelligence ( Volume: PP, Issue: 99 , July 2016 )

## 1. Problem statement

Content based image retrieval (CBIR) has become an active topic in the last decades. One application of interest is to implement the captioning of the image automatically after extraction of image features (encoding) and apply the sequential model to generate sentence description from the features (decoding).

Recently, convolutional neural network (CNN) has become a powerful tool for tackling on the image classification problems. By training multiple layers of convolutional filters, CNNs are able to learn the complex features of the images automatically and demonstrated superior performance compared to the low-level features. LSTMs (long short term memory) which are generally used for sequence modeling of tasks, will be trained as a language model conditioned on image encoding.

In this project, the objective is to generate the descriptions of input images.

## 2. Methodologies

The approach is divided into encoding the image with CNN and decoding the features with long-short term memory (LSTM) to generate natural language description.

#### 2.1 Encode - Convolutional neural network

Generally the goal is applicable to various convolutional neural network architectures. There are multiple outstanding deep neural networks: GoogLeNet, OxfordNet, Tensorflow, etc.

#### 2.2 Decode - Recurrent neural network

An “encoder” RNN reads the source sentence and transforms it into a rich fixed-length vector representation, which in turn in used as the initial hidden state of a “decoder” RNN that generates the target sentence.

A particular form of recurrent nets, called LSTM, is introduced. The LSTM model is trained to predict each word of the sentence after it has seen the image as well as all preceding words as defined by a function.


## 3. Datasets and Experiments

MSCOCO[2] dataset will be used as the training data. Various test image datasets will be used for testing, which will be decided later.

The works will be within the scope of:

- the training process of RNN and the approaches to address the under-fitting and over-fitting.

- generation result analysis

- Evaluation and improvement


## References
[1] R.Kiros, R.Salakhutdinov, and R.S.Zemel, “Unifying visual semantic embeddings with multimodal neural language models,”
in Transactions of the Association for Computational Linguistics, 2015.

[2] T.-Y. Lin, M.Maire, S.Belongie, J. Hays, P.Perona, D.Ramanan,P.Dollar, and C. L.Zitnick, “Microsoft coco: Common objects in context,” arXiv:1405.0312, 2014.
