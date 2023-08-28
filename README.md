# Deep NN Watermarking


This repository contains an implementation of a Deep Neural Network 
watermarking technique for protecting image processing networks from 
surrogate model attacks. 
It learns to invisibly hide a watermark image in the outputs of the 
model M, which should be protected.
Embedding this watermark will then be learned by a surrogate model
SM, trained on imitating the behaviour of M, such that the watermark
can be reconstructed from the outputs of the surrogate model. 

It is based on the method proposed by Zhang et al. 
"[Deep Model Intellectual Property Protection via Deep Model 
Watermarking](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9373945&tag=1)" 
(TPAMI 2021).
Although being based on the above method, this implementation differs
in many aspects from the original implementation. 


## Usage

---

To execute the files in this repository, clone the repository, set
`Deep_NN_watermarking` as your working directory and execute the 
following commands. 

Depending on your needs, you may want to execute only specific parts
of the provided code.
* Execute everything to apply the watermarking method on a model trained
on deblurring images
* To train only the watermarking method, execute only step 3
* If you want to apply this method to your own model, train your own 
model to be protected M and surrogate model SM in place of step 2 and 4


### 1. Preprocessing
Applies preprocessing to a directory `<dataset_path>` containing images 
such that it can be used for training the models below.
```
python datasets/preprocessing.py --data_path <dataset_path>
```

<br>

### 2. Deblurring
Trains a simple CNN on image deblurring. This is an exemplary task the 
model to be protected will learn. However, the watermarking method is 
applicable to any image processing task and this step can be 
substituted by training the model you want to protect. 

For deblurring, provide the train and validation data paths to the 
two datasets created in the first step.
```
python deblurring/main.py --blur_path_train <dataset path> --truth_path_train <dataset path> --blur_path_valid <dataset path> --truth_path_valid <dataset path>
```

<br>

### 3. Image Watermarking
In the next step, a hiding network H and reconstruction network R are
trained alongside a discriminator d on embedding on reconstructing 
the watermark image. 

First, create a dataset from the outputs of the previously trained 
model. 
```
python datasets/create_dataset.py --input_path <input dataset path> --save_path <output dataset path> --cnn_ckpt_path <model checkpoint path>
```

Next, train the watermarking models on the train and validation 
datasets from above. By default, the watermark image provided in 
`image_watermarking/watermarks/flower_rgb.png` will be embedded. 
```
python image_watermarking/main.py --train_data <dataset path> --valid_data <dataset path>
```

<br>

### 4. Surrogate Model Training
Trains a surrogate model on the watermarked outputs of the model 
to be protected.

First, again create the ground truth dataset by applying the 
hiding network H on the outputs of M, by providing a checkpoint
of model H and the dataset created in step 3 as input. 
```
python datasets/create_dataset.py --input_path <input dataset path> --save_path <output dataset path> --hnet_ckpt_path <model checkpoint path>
```

Then, train the surrogate model SM similarly to the model M. 
To simulate a realistic attack a different network architecture
and image dataset should be used. 
```
python deblurring/main.py --blur_path_train <dataset path> --truth_path_train <dataset path> --blur_path_valid <dataset path> --truth_path_valid <dataset path> --model 'conv'
```

<br> 

### 5. Reconstruction Network Finetuning
As a last step, the reconstruction network R is finetuned
on the surrogate model outputs. 

Create a dataset from the surrogate model trained above. 
```
python datasets/create_dataset.py --input_path <input dataset path> --save_path <output dataset path> --conv_ckpt_path <model checkpoint path>
```

Next, perform finetuning on a provided reconstruction 
network checkpoint. 
```
python finetuning/main.py --ckpt_path <model checkpoint path> --wmark_sm_path_train <dataset path> --wmark_sm_path_valid <dataset path> --wmark_h_path_train <dataset path> --wmark_h_path_valid <dataset path> --clear_blurred_path_train <dataset path> --clear_blurred_path_valid <dataset path> --clear_deblurred_path_train <dataset path> --clear_deblurred_path_valid <dataset path> 
```


## Requirements 

---

This implementation was tested on an environment with \
Python >= 3.10 \
PyTorch >= 2.0.1


## Licensing 

---

...
