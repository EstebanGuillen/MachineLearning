#using pandas DataFrame to hold training, testing, and validation data
import pandas as pd
import math

print("")

names_of_attributes = names=["label","cap-shape","cap-surface","cap-color","bruises","oder",
                                "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
                                "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
                                "stalk-color-above-ring","stalk-color-below-ring","veil-type",
                                "veil-color","ring-number","ring-type","spore-print-color",
                                "population","habitat"]

#read training data into pandas DataFrame
training_data = pd.read_table("data/training.txt", sep=",", names=names_of_attributes)

#read testing data into pandas DataFrame
testing_data = pd.read_table("data/testing.txt", sep=",", names=names_of_attributes)


#represents a decision node in the tree
class Node:
    #identifies which attribute this node tests on
    decision_attribute = ""
    
    #initialize node
    def __init__(self):
        self.decision_attribute = ""
        self.branches = {}
        
    #adds a child Node or Leaf 
    def add_branch(self,attribute_value,node):
        self.branches[attribute_value] = node
    
    #returns all branches
    def get_branches(self):
        return self.branches

#represents a leaf node in the tree
class Leaf:
    #label representing the classification value (e or p)
    label = ""

#global variables for the positive and negative label values (e and p)    
positive_value = "e"
negative_value = "p"

#global variable for setting the evaluation criteria (entropy or misclassification)
#can be set just before calling the id3 algorithm below
evaluation_criteria = "entropy"

#look-up table for the chi-square values (source https://dl.dropboxusercontent.com/u/63267778/chi_square_table.txt)
#key = dof, values[0] = alpha 0.5, values[1] =  alpha 0.05, values[2] = alpha 0.01
chi_square_look_up = {
                    1: [0.4549,3.841,6.635],
                    2: [1.3863,5.991,9.21],
                    3: [2.366,7.815,11.345],
                    4: [3.3567,9.488,13.277],
                    5: [4.3515,11.07,15.086],
                    6: [5.3481,12.592,16.812],
                    7: [6.3458,14.067,18.475],
                    8: [7.3441,15.507,20.09],
                    9: [8.3428,16.919,21.666],
                    10: [9.3418,18.307,23.209],
                    11: [10.341,19.675,24.725],
                    12: [11.3403,21.026,26.217],
                    13: [12.3398,22.362,27.688],
                    14: [13.3393,23.685,29.141],
                    15: [14.3389,24.996,30.578],
                    16: [15.3385,26.296,32],
                    17: [16.3382,27.587,33.409],
                    18: [17.3379,28.869,34.805],
                    19: [18.3377,30.144,36.191],
                    20: [19.3374,31.41,37.566]
                    }

#look-up table for the valid values of each attribute
valid_values = {"label": ["p","e"],
                "cap-shape": ["b","c","x","f","k","s"],
                "cap-surface": ["f","g","y","s"],
                "cap-color": ["n","b","c","g","r","p","u","e","w","y"],
                "bruises": ["t","f"],
                "oder": ["a","l","c","y","f","m","n","p","s"],
                "gill-attachment": ["a","d","f","n"],
                "gill-spacing": ["c","w","d"],
                "gill-size": ["b","n"],
                "gill-color": ["k","n","b","h","g","r","o","p","u","e","w","y"],
                "stalk-shape": ["e","t"],
                "stalk-root": ["b","c","u","e","z","r","?"],
                "stalk-surface-above-ring": ["f","y","k","s"],
                "stalk-surface-below-ring": ["f","y","k","s"],
                "stalk-color-above-ring": ["n","b","c","g","o","p","e","w","y"],
                "stalk-color-below-ring": ["n","b","c","g","o","p","e","w","y"],
                "veil-type": ["p","u"],
                "veil-color": ["n","o","w","y"],
                "ring-number": ["n","o","t"],
                "ring-type": ["c","e","f","l","n","p","s","z"],
                "spore-print-color": ["k","n","b","h","r","o","u","w","y"],
                "population": ["a","c","n","s","v","y"],
                "habitat": ["g","l","m","p","u","w","d"]
                }

#calculates the degrees of freedom for an attribute
def degress_of_freedom(attribute):
    return len(valid_values[attribute]) - 1

#returns the chi-square value given an attribute and alpha value
def chi_square_value_for_attribute(attribute,alpha):
    values = chi_square_look_up[degress_of_freedom(attribute)]
    if alpha == '0.5':
        return values[0]
    elif alpha == '0.05':
        return values[1]
    elif alpha == '0.01':
        return values[2]
    else:
        return 0.0
    
#returns the size (number of rows) of a DataFrame
def size(data):
    return len(data.index)

#returns a subset of a DataFrame, filtering on the value of an attribute
def filter_data(examples,attribute, value):
    return examples[examples[attribute] == value]
    
#tests if all the elements in the DataFrame (examples) have same value for an attribute    
def all_one_value(examples, attribute, attribute_value):
    filtered_examples = filter_data(examples, attribute, attribute_value)
    return size(filtered_examples) == size(examples)

#returns all possible values for an attribute
def possible_range_of_values(attribute):
    return valid_values[attribute]

#returns the most common value present the the examples for a given attribute
def extract_most_common_value(examples, attribute):
    values = possible_range_of_values(attribute)
    most_common = ""
    highest_count = 0
    for value in values:
        filtered = filter_data(examples, attribute, value)
        size_filtered = size(filtered)
        if size_filtered > highest_count:
            highest_count = size_filtered
            most_common = value  
    return most_common

#sub calculation for chi-square, calculates the chi-square for an attribute value
def chi_square_calculation_attribute_v(examples_v, target_attribute,attribute_v):
    total_count = size(examples_v)
    #if the size of examples_v is zero then there is nothing to calculate and return 0
    if total_count == 0:
        return 0.0
    
    positive_filtered_examples = filter_data(examples_v, target_attribute, positive_value)
    negative_filtered_examples = filter_data(examples_v, target_attribute, negative_value)
    
    observed_positive = size(positive_filtered_examples)
    expected_positive = (total_count/2.0)
                
    observed_negative = size(negative_filtered_examples)
    expected_negative = (total_count/2.0)
    
    pos = ((observed_positive - expected_positive)**2)/expected_positive
    neg = ((observed_negative - expected_negative)**2)/expected_negative
    return pos + neg
          
#returns the chi-square value for an attribute    
def chi_square_calculation(examples, target_attribute,attribute):
    chi_square_sum = 0.0
    values = possible_range_of_values(attribute)
    for value in values:
        examples_v = filter_data(examples, attribute,value)
        chi_square_v = chi_square_calculation_attribute_v(examples_v,target_attribute,value)
        chi_square_sum = chi_square_sum + chi_square_v
    return chi_square_sum
    
#calculates the entropy on a subset (examples_v) of data
def calculate_entropy(examples_v,target_attribute):
    entropy = 0.0
    total_size = size(examples_v)
    #if there is no data just return a zero value (does not contribute to calculation)
    if total_size == 0:
        return 0.0
    positive_filtered_examples_v = filter_data(examples_v, target_attribute, positive_value)
    negative_filtered_examples_v = filter_data(examples_v, target_attribute, negative_value)
    
    num_positive = size(positive_filtered_examples_v)
    calc_positive = 0.0
    if num_positive != 0:
        calc_positive = -(num_positive/total_size)*math.log((num_positive/total_size),2)
            
        
    num_negative = size(negative_filtered_examples_v)
    calc_negative = 0.0
    if num_negative != 0:
        calc_negative = - (num_negative/total_size)*math.log((num_negative/total_size),2)
        
    #returns -p(+)log p(+) - p(-)log p(-)    
    return calc_positive + calc_negative
   
#calculates the misclassification error on a subset (examples_v) of data
def calculate_misclassification_error(examples_v, target_attribute):
    error = 0.0
    total_size = size(examples_v)
    #if there is no data just return a zero value (does not contribute to calculation)
    if total_size == 0:
        return 0.0

    positive_filtered_examples_v = filter_data(examples_v,target_attribute,positive_value)
    negative_filtered_examples_v = filter_data(examples_v,target_attribute,negative_value)
    
    num_positive = size(positive_filtered_examples_v)
    p_positive = (num_positive/total_size)
                
    num_negative = size(negative_filtered_examples_v)
    p_negative = (num_negative/total_size)
    
    error = 1.0 - max([p_positive,p_negative])
    #returns 1 - max(probability of positive,probability of negative)
    return error
    
#returns the attribute with the best informatin gain using entropy as the impurity measure
def determine_best_attribute_entropy(examples, target_attribute, attributes):
    best_attribute = ""
    max_information_gain = 0.0
   
    #find the entropy of S (examples)
    entropy_s = calculate_entropy(examples,target_attribute)
    size_of_s = size(examples)
    
    
    #loop through attributes and calculate the gain
    for attribute in attributes:
        values = possible_range_of_values(attribute)
        information_gain = entropy_s
        for value in values:
            examples_v = filter_data(examples,attribute,value)
            entropy_v = calculate_entropy(examples_v,target_attribute)
            size_of_examples_v = size(examples_v)
            weight = (size_of_examples_v/size_of_s)
            information_gain = information_gain - weight*entropy_v
        #if we have a new candidate for the best attribute store the gain and attribute name
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute
             
    return best_attribute

#returns the attribute with the best gain using misclassification error as the impurity measure
def determine_best_attribute_misclassification_error(examples, target_attribute, attributes):
    best_attribute = ""
    max_gain = 0.0
    #find the misclassification error of S (examples)
    error_s = calculate_misclassification_error(examples,target_attribute)
    size_of_s = size(examples)
    
    #loop through attributes and calculate the gain
    for attribute in attributes:
        values = possible_range_of_values(attribute)
        gain = error_s
        for value in values:
            examples_v = filter_data(examples,attribute,value)
            error_v = calculate_misclassification_error(examples_v,target_attribute)
            size_of_examples_v = size(examples_v)
            weight = (size_of_examples_v/size_of_s)
            gain = gain - weight*error_v
        #if we have a new candidate for the best attribute store the gain and attribute name
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
       
    return best_attribute

#performs the id3 decision tree algorithm (recursive), returns the root node of the tree
#  examples - set (pandas DataFrame) of training examples provided for current iteration of the algorithm
#  target_attribute - the name of the column that provides the classification label
#  attribute_list - set of the names of the attributes for the training examples (excludes target_attribute)
#  alpha - the alpha value ["0.01","0.05","0.5","1.0"] used for chi-square pruning 
def id3(examples, target_attribute, attribute_list, alpha):
    root = Node()
    if(all_one_value(examples, target_attribute, positive_value)):
        leaf = Leaf()
        leaf.label = positive_value
        return leaf
    if(all_one_value(examples, target_attribute, negative_value)):
        leaf = Leaf()
        leaf.label = negative_value
        return leaf
    if(len(attribute_list) == 0):
        leaf = Leaf()
        leaf.label = extract_most_common_value(examples,target_attribute)
        return leaf
    #evaluate node selection using entropy or misclassification error
    if evaluation_criteria == 'entropy':
        a = determine_best_attribute_entropy(examples, target_attribute, attribute_list)
    else:
        a = determine_best_attribute_misclassification_error(examples, target_attribute, attribute_list)
    
    #for pruning get the calculated and threshold chi-square value
    chi_square_calculated = chi_square_calculation(examples,target_attribute,a)
    chi_square_lookup = chi_square_value_for_attribute(a, alpha)
    
    #if the calculated value is less than the threshold value then prune (decision node -> leaf node)
    if chi_square_calculated < chi_square_lookup:
        leaf = Leaf()
        leaf.label = extract_most_common_value(examples,target_attribute)
        return leaf
    root.decision_attribute = a
   
    values = possible_range_of_values(a)
    #iterate over all the possible values of an attribute and create branches for each value
    for value in values:
        root.branches[value] = ""
        examples_v = filter_data(examples,a,value)
        size_of_examples_v = size(examples_v)
        #if there are no more examples to train on create a leaf node for this branch
        if size_of_examples_v == 0:
            leaf = Leaf()
            leaf.label = extract_most_common_value(examples,target_attribute)
            root.add_branch(value,leaf)
        else:
            #remove attribute a so we don't reprocess it
            if a in attribute_list:
                attribute_list.remove(a)
            #add the next best decision node or leaf
            root.add_branch(value, id3(examples_v,target_attribute,attribute_list, alpha))
    return root



#traverses the decision tree one level and returns the node or leaf at the end of the selected branch
#  example_attribute_values - the example's attribute values <v1,v2,...vn>
#  node - decision node
def split_child(example_attribute_values, node):
    #from the example get the value of the attribute associated with is node example
    example_decision_attribute_value = example_attribute_values[node.decision_attribute]
    #return the node or leaf of the branch associated with the example's value of the decision attribute 
    return node.get_branches()[example_decision_attribute_value]

#classify an example, return the value of the first leaf node discovered
#  example_attribute_values - the example's attribute values <v1,v2,...vn>
#  node - decision node
def classify(example_attribute_values, node):
    if type(node) is Leaf:
        return node.label
    else:
        return classify(example_attribute_values, split_child(example_attribute_values,node))
    
#returns the accuracy of classifing on a testing data set
#  testing_data - the data set to classify (pandas DataFrame)
#  id3_root - the root of the id3 decision tree
#  print_statement - the string to print
def calculate_accuracy(testing_data, id3_root, print_statement):
    count_correct = 0
    count_false = 0
    accuracy = 0.0
    for i,r in testing_data.iterrows():
        #if the classifing label in the data set matches the calculated one we classified correctly
        if r["label"] == classify(r,id3_root):
            count_correct = count_correct + 1
        else:
            count_false = count_false + 1
    if count_correct > 0:
        accuracy = (count_correct)/(count_correct + count_false)  * 100
    
    print(print_statement, accuracy, "%")
    print("  ", "Number correctly classified:   ", count_correct) 
    print("  ", "Number incorrectly classified: ", count_false) 
    return accuracy
    

#helper function for getting the list of attributes for the data sets
def get_attributes():
    
    #list of attributes to be used in the id3 algorithm. 
    #After an attribute is selected for a node of the tree it is removed from the list
    attributes = ["cap-shape","cap-surface","cap-color","bruises","oder",
                "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
                "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
                "stalk-color-above-ring","stalk-color-below-ring","veil-type",
                "veil-color","ring-number","ring-type","spore-print-color",
                "population","habitat"]
    return attributes

best_accuracy = 0.0
best_tree = Node()
best_tree_name = ''

#the following group of statements run the id3 algorithm for different values of evaluation criteria and alpha (for pruning)
evaluation_criteria = "entropy"
alpha = '1.0'
print_statement = "Accuracy (using entropy and confidence level of 0):  "
id3_entropy = id3(training_data,"label",get_attributes(),alpha)
accuracy = id3_entropy_accuracy = calculate_accuracy(testing_data,id3_entropy, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_entropy
    best_tree_name = 'id3_entropy_confidence_level_0'

evaluation_criteria = "entropy"
alpha = '0.01'
print_statement = "Accuracy (using entropy and confidence level of 99): "
id3_entropy_alpha_01 = id3(training_data,"label",get_attributes(),alpha)
accuracy = id3_entropy_alpha_01_accuracy= calculate_accuracy(testing_data,id3_entropy_alpha_01, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_entropy_alpha_01
    best_tree_name = 'id3_entropy_confidence_level_99'

evaluation_criteria = "entropy"
alpha = '0.05'
print_statement = "Accuracy (using entropy and confidence level of 95): "
id3_entropy_alpha_05 = id3(training_data,"label",get_attributes(),alpha)
accuracy = id3_entropy_alpha_05_accuracy = calculate_accuracy(testing_data,id3_entropy_alpha_05, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_entropy_alpha_05
    best_tree_name = 'id3_entropy_confidence_level_95'

evaluation_criteria = "entropy"
alpha = '0.5'
print_statement = "Accuracy (using entropy and confidence level of 50):  "
id3_entropy_alpha_5 = id3(training_data,"label",get_attributes(),alpha)
accuracy = id3_entropy_alpha_5_accuracy = calculate_accuracy(testing_data,id3_entropy_alpha_5, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_entropy_alpha_5
    best_tree_name = 'id3_entropy_confidence_level_50'


print("")

evaluation_criteria = "misclassification"
alpha = '1.0'
print_statement = "Accuracy (using misclassification and confidence level of 0):  "
id3_misclassification_error = id3(training_data,"label",get_attributes(),alpha)
accuracy = calculate_accuracy(testing_data,id3_misclassification_error, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    print(accuracy, best_accuracy)
    best_tree = id3_misclassification_error
    best_tree_name = 'id3_misclassification_error_confidence_level_0'

evaluation_criteria = "misclassification"
alpha = '0.01'
print_statement = "Accuracy (using misclassification and confidence level of 99): "
id3_misclassification_error_alpha01 = id3(training_data,"label",get_attributes(),alpha)
accuracy = calculate_accuracy(testing_data,id3_misclassification_error_alpha01, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_misclassification_error_alpha01
    best_tree_name = 'id3_misclassification_error_confidence_level_99'

evaluation_criteria = "misclassification"
alpha = '0.05'
print_statement = "Accuracy (using misclassification and confidence level of 95): "
id3_misclassification_error_alpha05 = id3(training_data,"label",get_attributes(),alpha)
accuracy = calculate_accuracy(testing_data,id3_misclassification_error_alpha05, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_misclassification_error_alpha05
    best_tree_name = 'id3_misclassification_error_confidence_level_95'

evaluation_criteria = "misclassification"
alpha = '0.5'
print_statement = "Accuracy (using misclassification and confidence level of 50):  "
id3_misclassification_error_alpha5 = id3(training_data,"label",get_attributes(),alpha)
accuracy = calculate_accuracy(testing_data,id3_misclassification_error_alpha5, print_statement)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = id3_misclassification_error_alpha5
    best_tree_name = 'id3_misclassification_error_confidence_level_50'
    

print("")    
print("best tree: ", best_tree_name)

print("")



#read validation data into pandas DataFrame    
validation_data = pd.read_table("data/validation.txt", sep=",", names=names_of_attributes)


import os
remove_chars = len(os.linesep)


file = open("validation-best-accuracy.txt", "w")
for i,r in validation_data.iterrows():
    file.write(classify(r,best_tree) + "\n")

file.truncate(file.tell() - remove_chars)
file.close()

print("classification predictions on validation data written to: validation-best-accuracy.txt")

print("")

