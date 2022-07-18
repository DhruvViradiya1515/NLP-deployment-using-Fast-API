# NLP-deployment-using-Fast-API
## Problem Statement
* Airline sentimental dataset is provided. The task is to perform natural language processing(NLP) and create an API for final model
> Dataset: [link](https://drive.google.com/file/d/1iHdXv0ex90AT3T2JqFlTRqNtZuATkEJn/view?usp=sharing)
## Prerequisites
* NLP(natural language processing)
* Sci-kit learn
* FastAPI

## Dependencies
* sklearn
* nltk
* fastapi
* numpy, pandas, matplotlib

## Approach
* Cleaning the dataset: 
    * Removing special characters, emojis and extra words.
    * Removing stop words
    * lemmetizing the words
    
* Training:
    * Using vectorizor for making tranining dataset
    * Applying PCA for reducing dimensions
    * Train different model and compare their results.
* Generating API
    * Using fastapi for creating api for generated model.

## Accuracy-score for different models:
 | Model | Accuracy-score |
 | :---: | :---:|
 | Decision Tree | 0.8423 |
 | Random Forest | 0.8847 |
 | XGBoost | 0.87 |
 | Multinomial NB | 0.8310 |
 | Light GBM | 0.8930 |
 | Gradient Boosting | 0.8739 |
 | Logistic Regression | 0.9073 |

## ROC plot for all the models:
![alt text](./images/Roc_all.png)
