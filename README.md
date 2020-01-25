# Skyline Image Generation
This project is a PyTorch implementation of [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.

#### Model Architecture
##### Generator
![Alt text](example/generator.png?raw=true "Title")
##### Discriminator
![Alt text](example/discriminator.png?raw=true "Title")

## Goal & Motivation
This is a course project (APS360) from University of Toronto. Our main purpose is to generate city skyline images from input sketch. If interested, you can take a look at our project report for more details. 

#### Sample Result
![Alt text](example/output_combined.png?raw=true "Title")

## Data
#### Source
We collected images from Flickr, an image hosting service. In case you want to view our dataset, the download link is [Google Drive](https://drive.google.com/file/d/1F6Y4xk3wK-MYrir3G0jP0ev9xEeYAdlo/view?usp=sharingn). This 'data_uncleaned.zip' should contain our uncleaed images downloaded directly from Flickr. We also provide our cleaned dataset here [Google Drive](https://drive.google.com/file/d/1Z4OJtYJWkbydLAIpiJ8ysvSar_3lp-5F/view?usp=sharing). 'input_edges' contains input sketches generated from [gPb](https://github.com/vrabaud/gPb) and 'real' has the corresponding real skyline images.

## Usage
#### Train on your own
You can collect your own skyline images and obtain building boundaries by using gPb. Then you are able to train your own model with [train.py](train.py). Feel free to change the model architecture in [model.py](model.py), and change any hyperparameters we set in train.py.

#### Use Pre-trained Model
Our team also provides our pre-trained model so that people can generate images directly without training. Our model isn't the best, but it should give you some fine results.

## Demo
The following is the instruction on how to generate a skyline image from a sample [test.jpg](demo/test/test/test.jpg).

#### Setting Up Environment
We use Anaconda with python version 3.6.7. Pytorch is the main framework we are using. Please install pytorch before running the demo code
#### Step-By-Step instructions 
1. Download our pre-trained model from [Google Drive](https://drive.google.com/file/d/15jCidE6pM2cnIuAL4wJbd_NR12YhTXKi/view?usp=sharing), and put this model file under the "demo" directory.
2. Go to the "demo" directory and run demo.py.
```
cd demo
python demo.py
```
3. The output message should be 
```
Done generating images, please check 'output' and 'comparison' directory.
```
4.  All Done! A generated image is stored under the 'output' directory. This is the output from the previous test.jpg.

## Acknowledgments
Our model is inspired by [pix2pix](https://arxiv.org/abs/1611.07004). The pytorch version of the code is [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Also I need to say thanks to my teammates, [Tom](https://github.com/luoshuya) and [Jeremy](https://github.com/jeremyxu1998). It has been a great pleasure working with you!


