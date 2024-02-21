<img src="figures/header.jpg" style="display:block; margin:auto; width:75;">

<br><br> 

<h1>
<p align = "center" background-color = "darkgray">
            <strong>Dog Breed Image Classification</strong>
</p>  
</h1>
<h2>
<p align = "center">
    End-to-end Deep Learning Project for
    <br>Dog Breed Multi-Class Image Classification:
    <br>EDA, Transfer Learning, Fine-Tuning, and Model Deployment
</p>
</h2>

# 1. Introduction

## Background

In recent years, the canine population has witnessed a surge in the number of mixed-breed dogs. While the charm of mixed-breed dogs lies in their one-of-a-kind characteristics, the increasing variety poses challenges when it comes to identifying their specific breeds. This is where a classification model proves to be invaluable.

Creating a classification model for identifying breeds of dogs can serve various purposes including:
- Lost and found services to help efficiently reunite lost dogs with owners.
- Educational tools through interactive websites or mobile applications to learn about different breeds.
- Pet adoption platforms to identify breeds within shelters, enhancing adoptiong and rehoming processes.
- Dog shows and competitions to assist judges in verifying participating dog breeds.

In the [modeling notebook](notebooks/modeling_notebook), you will explore the multi-class [stanford dogs image dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset), then uses transfer learning to train and fine-tune four modern neural network architectures. We'll use built-in [Keras applications](https://keras.io/api/applications) to use architectures with pre-trained ImageNet weights for the following models: 
- [xception](https://arxiv.org/abs/1610.02357)
- [InceptionResNetV2](https://arxiv.org/abs/1602.07261)
- [EfficientNetB3](https://arxiv.org/abs/1905.11946)
- [ConvNeXtSmall](https://arxiv.org/abs/2201.03545)
**WARNING**: Do not attempt to run the entire notebook without a GPU with at least 8GB of memory. The models may take hours or even days to train.

We saved the best model and deployed it two ways in the [deployment notebook](notebooks/deployment_notebook), with a front-end application hosted by streamlit.
- **Serverless Deployment**: Save the best model with tensorflow lite, create a docker image for the environment with minimized dependencies, and deploy it to the cloud using AWS Lambda
- **Kubernetes Deployment**: Convert the model to tensorflow format, set up a flask application, build docker images for them, and deploy the full application to AWS EKS. 
- **Front-end Application**: Create a low-code front-end for our application using streamlit and connect it to the cloud-deployed AWS services.

The full framework for that you will complete through both notebooks for this project is shown below:

<img src="figures/dog-prediction-framework.png" width="700">

## Table of Contents.

The table of contents for the repository is as follows.

### 1. Introduction
- Description of the project methodology and real world use-case.
- Table of contents **(We Are Here)**
- Repository structure
- Dataset source and information.
### 2. Prerequisites
- Software
- Clone the repository
- Download data
- Local conda environment setup, including tensorflow installation.
### 3. Analysis
- EDA
- Tune Xception, InceptionResNetV2, EfficientNetB3, ConvNeXtSmall
- Comparison and analysis of all four tuned models to determine the final model: EfficientNetB3V2
### 4. Serverless Deployment
- Local deployment with Docker
- Cloud deployment with AWS Lambda
### 5. Kubernetes Deployment
- Local deployment with Docker
- Cloud deployment with AWS EKS
### 6. Application Deployment
- Application deployment with Streamlit
### 7. References
- References for all model architectures and Keras documentation


## Repository Structure

The following details the files stored in this repository. Note that the inception and convnext models were excluded due to large file size.

```
dog-prediction
│   README.md
│   requirements.txt - Set up local environment for running notebook
│
└───data - Unzip the dataset to this folder (section 2.3)
│   
└───figures - Contains images for the notebook and readme
│
└───models
│   │   effnetV2B3_model.keras
│   │   Xception_model.keras
│   │   model.tflite - Tflite model with minimized dependencies for serverless deployment
│   │   converted_model - Tensorflow model for kubernetes deployment with tf-serving
│
└───notebooks
│   │   modeling_notebook.ipynb - EDA, modeling and hyperparameter tuning
│   │   deployment_notebook.ipynb - Local, AWS Lambda and AWS EKS deployment with docker, kubernetes and flask
│   │   convnext.py - Hotfix for saving convnext models with tensorflow 2.10
│
└───python
│   │   gateway.py - Flask application that connects to tf-serving model
│   │   lambda_function.py - Function for AWS Lambda cloud deployment
│   │   proto.py - Converts model to proto for efficient deployment and inference
│   │   streamlit_app.py - Application for streamlit cloud
│   │   test.py - Tests local, serverless, and cloud kubernetes inference
│
└───deployment
    │   docker-compose.yaml - Docker Compose file for setting up multi-container network
    │   eks-config.yaml - Configuration file for building EKS cluster
    │   lambda.dockerfile - Docker environment with tflite for lambda deployment
    │
    └───gateway
    │   │   gateway-deployment.yaml - Kubernetes file for deploying the gateway with flask
    │   │   gateway-service.yaml - Kubernetes file for serving the gateway to the model (port 80 -> 9696) 
    │   │   image-gateway.dockerfile - Dockerfile for running gateway container to post predictions to the model
    │   │   Pipfile - Pipfile for building gateway container environment
    │   │   Pipfile.lock - Pipfile lock for building gateway container environment
    │
    └───model
        │   model-deployment.yaml - Kubernetes file for deploying the model with tensorflow-serving     
        │   model-service.yaml - Kubernetes file for serving the model (port 8500) 
        │   image-model.dockerfile - Dockerfile for running model container to accept posts for predictions
```

## Dataset Source

The Stanford Dogs dataset contain images of dogs from around the world. It is primarily used for image categorization of dog breeds. The version for this project can be downloaded from [Kaggle - Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset).
- Dog breeds: 120
- Number of images: 20,580
- This project uses the "Images" folder which is divided into subfolders by class (dog breed)

Further observation of the data can be found in section 3 EDA

# 2. Prerequisites

## Software

```
- Docker
    - [Official installation documentation](https://docs.docker.com/engine/install/) for Docker Engine.
- Anaconda or Miniconda
    - [Official installation documentation](https://docs.anaconda.com/free/anaconda/install/index.html). Install with CLI tools.
- Python 3.10
    - This should be installed with the environment setup, but note the [official python download page](https://www.python.org/downloads/) for reference.
- Tensorflow 2.10
    - This should be installed with the environment setup, but installation instructions could vary depending on OS. Please refer to the [official documentation](https://www.tensorflow.org/install/pip) for your machine.
```

## Clone the repository
```
git clone https://github.com/PriyaVellanki/flower_classification.git
```

## Data download

Download the dataset from [Kaggle - Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset). Extract the contents of the zip file to the ``data`` folder. Ensure that the folder structure is as follows before proceeding.

```
data
│
└───annotations
│   │
│   └──── Annotation
└───images
    │
    └──── Images
```

## Local conda environment setup, including tensorflow installation.

The following will create a conda environment with the necessary requirements for this project.

``conda create --name dog-prediction python=3.10``

``conda activate dog-prediction``

``pip install -r requirements.txt``

This project was created with tensorflow 2.10.0 with GPU support for windows cuda tool kit 11.2 and nvidia cuda-nvcc. The tensorflow installation for your OS may be different. Please refer to the [official documentation](https://www.tensorflow.org/install/pip). 

For a windows build with a cuda-enabled nvidia graphics card, please run the following commands.

``conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0``

``conda install -c nvidia cuda-nvcc``


# 3. Directions

The EDA, data preparation, model tuning and comparison were completed in the [modeling notebook](notebooks\modeling_notebook). 
The prerequisites from the previous section should prepare you to follow along with the notebook.

## EDA

To explore the data, we first looked at the full list of classes, or dog breeds, stripped from the folder structure names.

```
List of categories =  ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black and tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Wire haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft coated wheaten terrier', 'West highland white terrier', 'Lhasa', 'Flat coated retriever', 'Curly coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake bay retriever', 'German short haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old english sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater swiss mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great dane', 'Saint bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog'] 
```

We computed metrics for the maximum, minimum and mean number of images within a class, the kurtosis and skew of the classes within the dataset. The dataset is highly skewed, with a skewness > 1, and platykurtic with a kurtosis < 3.

```
Max counts: 166
Min counts: 88
Mean counts: 109.758
Kurtosis: 0.788
Skew: 1.0138674098824882
Top 5 breeds: ['Afghan hound' 'Maltese dog' 'Scottish deerhound' 'Shih tzu' 'Samoyed']
Bottom 5 breeds: ['French bulldog' 'Doberman' 'Brittany spaniel' 'Welsh springer spaniel'
 'Bouvier des flandres']
```

A barchart showing the distribution over all classes. The maximum limit has been marked with a bold line, the minimum limit with a dashed line. We can see a large distribution between classes.

<img src="figures/eda1.PNG">

The top 5 and bottom 5 classes by number of images. Examples for both groups are shown in the notebook, with the top 5 shown below.

<img src="figures/eda2.PNG">


## Modeling

The models Xception, InceptionResNetV2, EfficientNetB3V2 and ConvNeXtSmall were chosen for their small number of parameters and comparable performance metrics. We used transfer learning to tune each keras models with pre-trained weights from training on the ImageNet dataset. The function ``make_model`` in section 4.1 of the notebook has the following features:
  - Takes any [keras application](https://keras.io/api/applications/) model as input and freezes the convolution layers and weights trained on ImageNet
  - Adds a trainable pooling layer, trainable dense layer, trainable dropout layer and second dense layer after the base model
  - Learning rate, inner layer size and dropout rate can be adjusted for hyperparameter tuning on these layers

Before training, we split the dataset into training and test sets. The test set was held out for final performance calculations, and we tuned the model parameters using a once again separated training and validation set. The final split training/validation/testing split was 60/20/20. The notebook uses seeds for reproducibility of results.

For each model, we used validation curves to tune within the following parameter ranges:

- Learning rate in [0.00001, 0.0001, 0.001, 0.01, 0.1]
- Inner layer size in [10, 50, 100, 250, 500, 1000]
- Dropout rate in [0.25, 0.4, 0.5, 0.6, 0.75]

We then unfroze the first 4 layers and re-trained the model with the best performing hyperparameters on the held-out test set. Each of the best performing models are compared in the following notebook section.

## Model comparison

The following table compiles performance metrics for all four trained models. EfficientNetB3V2 was our best performing model across all metrics. It also has the smallest size and a relatively small number of parameters, so we can assume it will run fairly fast. It was converted to tflite and saved to [model.tflite](models/model.tflite) for efficient performance during deployment.

Model | Size | Parameters (k) | Train Time (s) | Inference Time (s) | Test accuracy | Test precision | Test recall | Test f1
--- | --- | --- | --- | --- | --- | --- | --- | ---
Xception | 82.333 | 21078 | 70.203 | 12.696 | 0.901 | 0.898 | 0.900 | 0.897
InceptionV2 | 210.371 | 11198 | 100.762 | 24.258 | 0.920 | 0.917 | 0.921 | 0.916
EfficientNetB3V2 | 54.715 | 13344 | 66.558 | 13.107 | 0.945 | 0.944 | 0.945 | 0.944
ConvNeXtSmall | 191.716 | 49677 | 265.549 | 160.441 | 0.939 | 0.938 | 0.940 | 0.937


# 4. Serverless Deployment
## Local

To deploy locally, navigate up to the root folder and run the following from a terminal or command prompt. Please note that docker must already be installed and running on your local machine. See the [official documentation](https://docs.docker.com/engine/install/) for Docker Engine installation.

The following commands will build an image from the [dockerfile](deployment/lambda.dockerfile) using python 3.10, tflite and [lambda_function.py](python/lambda_function.py). Once the container is running, test it.

**Build**
> docker build -t dog-prediction -f deployment/lambda.dockerfile .

**Run**
> docker run -it --rm -p 8080:8080 dog-prediction:latest

**Test**
> python python/test.py -t local

It should return the following response with the top 5 dog breeds ascending order, each followed by the corresponding model score.
```
[['Golden retriever', 6.056427955627441], ['Great pyrenees', 2.281074285507202], ['Kuvasz', 1.2518045902252197], ['Labrador retriever', 0.1173974946141243], ['Flat coated retriever', -1.8855047225952148]]
```

## AWS Lambda

The model is hosted serverlessly on an AWS Lambda function with a docker image hosted on AWS ECR. 
The steps for deploying the model to AWS Lambda are included in the [deployment notebook](notebooks/deployment_notebook), section 2.

Run the following command to test the serverless model. It is expected to return the same response as for the locally deployed image.

**Test**
> python python/test.py -t lambda

# 5. Kubernetes Deployment
## Local
To test our kubernetes deployment for the cloud, we first set up a multi-container environment with docker-compose. It references our best model saved to tensorflow format, stored in [model_converted](models/model_converted). 

To locally connect the gateway (port 9696) to the serving model (port 8500), spin up a virtual network using docker-compose.
- The docker compose file is saved to [docker-compose.yaml](deployment/docker-compose.yaml).

Build the images from the root folder, then **navigate to the deployment folder** then and run docker-compose. Follow by testing to ensure the returned predictions are as expected, then shut down the stack.

**Build**
> docker build -t dog-prediction-model:v1 -f deployment/model/image-model.dockerfile .
>
> docker build -t dog-prediction-gateway:v1 -f deployment/gateway/image-gateway.dockerfile .

**Run**
> docker-compose up

**Test**
> python python/test.py -t gateway

**Shut Down**
> docker-compose down


**Expected Predictions**
```
[['Golden retriever', 6.056427955627441], ['Great pyrenees', 2.281074285507202], ['Kuvasz', 1.2518045902252197], ['Labrador retriever', 0.1173974946141243], ['Flat coated retriever', -1.8855047225952148]]
```

## AWS EKS (Elastic Kubernetes Service)

The full steps for deploying the model to AWS EKS are included in the [deployment notebook](notebooks/deployment_notebook), section 3. 


Follow along with section 3.3 in the [deployment notebook](notebooks/deployment_notebook) to push the docker images AWS ECR and create pods and services, then spin up a cluster with the following command. Ensure that you are logged into a user account with sufficient permissions for deploying a cluster.

**NOTE**: The cost for deploying this cluster is ~$0.17/hour.

**Create**
> eksctl create cluster -f deployment/eks-config.yaml

**Shut down**
> eksctl delete cluster --name=dog-prediction-eks --disable-nodegroup-eviction


# 6. Application Deployment

The model is also deployed to an interactive application hosted by Streamlit. The application performs inference using the cloud-deployed model. Run ``streamlit run python/streamlit_app.py`` to test it locally, or go to [https://dog-prediction.streamlit.app/](https://dog-prediction.streamlit.app/) to use the deployed application.

- Click 'Browse files' to upload an image.
- Click 'Predict' to post to the cloud.
- While the image is being processed, you should see a status wheel.
- Once it has completed, the application will output a table with the top 5 dog breeds and model scores.

See below for an example:

<img src="figures/app.PNG">


# 7. References

1. Chollet, François (2016). _“xception: Deep Learning with Depthwise Separable Convolutions”_. In: CoRR abs/1610.02357. arXiv: 1610.02357. url: [http://
arxiv.org/abs/1610.02357](http://arxiv.org/abs/1610.02357).

2. Szegedy, Christian, Ioffe, Sergey, and Vanhoucke, Vincent (2016). _“Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning”_. In: CoRR abs/1602.07261. arXiv: 1602.07261. url: [http://arxiv.org/abs/1602.07261](https://arxiv.org/abs/1602.07261).
   
3. Sandler, Mark, Howard, Andrew G., Zhu, Menglong, Zhmoginov, Andrey, and Chen, Liang-Chieh (2018). _“Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation”_. In: CoRR abs/1801.04381. arXiv: 1801 . 04381. url: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381).
   
4. Tan, Mingxing, Chen, Bo, Pang, Ruoming, Vasudevan, Vijay, Sandler, Mark, Howard, Andrew, and Le, Quoc V. (2019). _MnasNet: Platform-Aware Neural Architecture Search for Mobile_. arXiv: [1807.11626 \[cs.CV\]](https://arxiv.org/abs/1807.11626).

5. Tan, Mingxing and Le, Quoc V. (2020). _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_. arXiv: [1905.11946 \[cs.LG\]](https://arxiv.org/abs/1905.11946).

6. Liu, Ze, Lin, Yutong, Cao, Yue, Hu, Han, Wei, Yixuan, Zhang, Zheng, Lin, Stephen, and Guo, Baining (2021). _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows_. arXiv: [2103.14030 \[cs.CV\]](https://arxiv.org/abs/2103.14030).

7. Liu, Zhuang, Mao, Hanzi, Wu, Chao-Yuan, Feichtenhofer, Christoph, Darrell, Trevor, and Xie, Saining (2022). _“A ConvNet for the 2020s”_. In: CoRR abs/2201.03545. arXiv: 2201 . 03545. url: [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

8. Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.

9. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.
    
10. Fu, Y. (2020, June 30). Keras documentation: Image Classification via fine-tuning with EfficientNet. [https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
