# Data Processing Final proyect GitHub Jon Lejardi Iker Aldasoro - Universidad Carlos III [![python version](https://img.shields.io/badge/python-3.12.6+-blue.svg)](https://www.python.org/downloads/)

This repository holds the files created for the final proyect of the subject Data Processing.

Here we will explain the steps followed for the completion of the final project and the results obtained. 
Basic Project
For the basic project we had to solve a regression task, comparing the performance of different vectorizations and machine learning strategies.

**1.	Input Variable Analysis**

Firstly, we are asked to visualize the relationship between the output variable, “rating”, and the input variable “categories”. Before starting with it, we imported all the libraries and packets needed for the project and did some preprocessing by deleting the null values from both variables.

After separating and counting all the existing categories, they were too many to do a proper analysis, so we had to limit the range. That is why we picked the 15 most common categories and do a box diagram to see the average rating and standard deviation of recipes with any of those categories. See figure bellow.

FIRST GRAPH

At a first glance, all the categories studied have a very high average rating, with the exception of a few poorly rated recipes. Otherwise, there are no conclusions to be drawn from this graph. Another, more exhaustive analysis would be necessary.
We also performed a correlation matrix of the integer input variables as “sodium”, “fat”, “calories”, “protein” against the output variables. But we can tell there is no correlation at all between them.
CORRELATION MATRIX
To complete this first analysis we carried out other studies comparing the text length to the recipe’s rating.
TEXT LENGTH vs RATING
And al

