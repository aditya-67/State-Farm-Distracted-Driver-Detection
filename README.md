## State-Farm-Distracted-Driver-Detection

### Dataset can be downloaded [here](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

### Methodology
* Transfer learning (VGG16 + CNN)
  
  > Initially, split our training data into train, test and validation. We use a VGG16 model trained on Imagenet, excluding the top fully-connected layers and run train, test and validation images through VGG to obtain its deep features (until last convolutional layer) and save them. We create a top_model which takes deep features of VGG as input and train the model by loading data from the directory and these features. Save these weights and test our model on the test data. 

### Solution
* Transfer Learning (VGG16 + CNN) - Kaggle Score (0.24259)
  > Please refer to Solution.ipynb for the optimal solution 
* Vanilla CNN (Please refer to folder cnn)
