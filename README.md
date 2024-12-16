# Data Processing Final proyect GitHub Jon Lejardi Iker Aldasoro - Universidad Carlos III [![python version](https://img.shields.io/badge/python-3.12.6+-blue.svg)](https://www.python.org/downloads/)

This repository holds the files created by Iker Aldasoro Marculeta and Jon Lejardi Jericó for the final proyect of the subject Data Processing.

Here we will explain the steps followed for the completion of the final project and the results obtained. 
Basic Project
For the basic project we had to solve a regression task, comparing the performance of different vectorizations and machine learning strategies.

**1.	Input Variable Analysis**

Firstly, we are asked to visualize the relationship between the output variable, “rating”, and the input variable “categories”. Before starting with it, we imported all the libraries and packets needed for the project and did some preprocessing by deleting the null values from both variables.

After separating and counting all the existing categories, they were too many to do a proper analysis, so we had to limit the range. That is why we picked the 15 most common categories and do box plot to see the mean rating and standard deviation of those categories.

![image](https://github.com/user-attachments/assets/91c347e0-7b68-4a64-b49d-f7f7f9e64b19)

At a first glance, we can tell that all the categories studied have a very high mean rating. Otherwise, there are no conclusions to be drawn from this graph. Another, more exhaustive analysis would be necessary.
We also performed a correlation matrix of the integer input variables as “sodium”, “fat”, “calories”, “protein” and the output variable. But we can see there is no correlation at all between them. Bellow they are represented in a plot separately, but we reach to the same conclusion.

![image](https://github.com/user-attachments/assets/aec09f72-3d57-475b-b3a5-d86943011000)

![image](https://github.com/user-attachments/assets/609f1b2e-1c92-4cfa-a45b-a9f68ab35670)

To complete the comparation of input variables with the output one, on the one hand we tried to see if there is any correlation between the text length and the recipe's rating, but it doesn't seem significant.

![image](https://github.com/user-attachments/assets/e1124ea3-e710-4507-b96f-16ed96d83d0d)

For the missing input variable "years", it is interesting to see the difference of rating variation. The recipes between 2004 and 2011 keep a high mean rating with a reasonable deviateino, while in 2003 and from 2012 in advance, the ratings are much more distributed.

![image](https://github.com/user-attachments/assets/5557bbd5-1934-49f0-aed7-cc88232ba7a9)

To complete this first analysis we counted the ratings and represented their distribution and outliers.x 

![image](https://github.com/user-attachments/assets/26c26047-9f4f-4029-a27f-21e6656af874)

 ![image](https://github.com/user-attachments/assets/6697e53b-0e35-4a63-9c84-9fd60686386b)

Here it is seen that the majority of the ratings are between 3 and 5, having the highes concentration in the range of 3.5 - 4.5. On the other side, recipes with bad ratings (under 2.5) are rare, but among the failed ones a rating of 0. is more common.

**2. Text Preprocessing**

In order to prepare the text data so it can be processed by the regression models it has to go through a multi stage processing composed by tokenization, homogeneization, cleaning and voectorization. In this second stage we perform the first three stages of the processing to keep the text variables ready to be transformed to numerical. 
At first the phrases are divided into tokens (words). Then starts the homogeneization, where all the verbs are lemmatized (conjugated verbs are changed to the basis) and everything is in lowercase. Finally the cleaning is done, here special characters, extra white spaces, stop words and other meaningless elements are removed.

**3. Vector Representation**

After the text preprocessing, so the machine learning models can work with the information in text variables, these variables are transformed into word vectors. This means, they will have a numerical representation where their value is related to the meaning they have.
For this type of transformation we are asked to use three different methods:

- TF-IDF: Gives a high value for a given term in a given document if that terms occurs often in that particular document and very rarely anywhere else. I just focuses on word importance, doesn't take into account the context. The size of the results obtained from this method is (20130, 1000), since we stablished a maximum of 1000 features.
  
- Word2Vec: It has a shallow word relationship, but lacks full contextual understanding. The size of the results is (20130, 100), since we stablished a maximum of 100 features for the vectorization. Along this parameters we used a window of 5 to create the mean vector.
  
- BERT: We were free to choose a transform-based model, so we chose BERT. It is context-aware and provides deep contextual embedings. The result size is (20130, 768), since 768 is the maximum size of the hidden layer, which is the length of the output embeddings.

**4. Training and Evaluation of Regression Models**











