Machine Learning Project 1

Description of Files:

id3.py - python code for id3 algorithm (all the code for project)

Project_1.ipynb - IPython Notebook containing all the code from id3.py and some additional queries on the data for validating folklore rules

Project_1.html - HTML export of IPython Notebook

README.md - markdown README for GitHub

validation-best-accuracy.txt - predictions for the validation test set of data. ***USE THIS FILE FOR GRADING***

data/training.txt - the training data used to build the decision trees 

data/testing.txt - the testing data used to test the accuracy of the decision trees

data/validation.txt - the hidden (labels not known) data used for validating our decision tree 

MachineLearningProjectReport.pdf - project report file, summary of code and results

MachineLearningProjectReport.docx - project report file, summary of code and results

***** RUNNING THE CODE ******

Tested on MacOS

Dependencies:
Python3
pandas 0.17.1

To run:

>python3 id3.py

Decision trees will be built from the training data.
Testing data will be used to evaluate each tree.
Accuracy values from each tree will be printed to the console.
Using the “best tree” the validation data will get classified and the results will be written to validation-best.accuracy.txt

