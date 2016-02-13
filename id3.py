import pandas as pd

from pandas import set_option

import math
set_option("display.max_rows", 10)

print("")

training = pd.read_table("data/training.txt", sep=",", 
                         names=["label","cap-shape","cap-surface","cap-color","bruises","oder",
                                "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
                                "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
                                "stalk-color-above-ring","stalk-color-below-ring","veil-type",
                                "veil-color","ring-number","ring-type","spore-print-color",
                                "population","habitat"])


testing_data = pd.read_table("data/testing.txt", sep=",", 
                             names=["label","cap-shape","cap-surface","cap-color","bruises","oder",
                            "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
                            "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
                            "stalk-color-above-ring","stalk-color-below-ring","veil-type",
                            "veil-color","ring-number","ring-type","spore-print-color",
                            "population","habitat"])


#represents a test node in the tree
class Node:
    decision_attribute = ""
    
    
    def __init__(self):
        self.decision_attribute = ""
        self.branches = {}
        
    def add_branch(self,attribute_value,node):
        self.branches[attribute_value] = node
    
    def get_branches(self):
        return self.branches

#represents a leaf node in the tree
class Leaf:
    label = ""
    
positive_value = "e"
negative_value = "p"
evaluation_criteria = ""

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

attributes_master = ["cap-shape","cap-surface","cap-color","bruises","oder",
                "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
                "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
                "stalk-color-above-ring","stalk-color-below-ring","veil-type",
                "veil-color","ring-number","ring-type","spore-print-color",
                "population","habitat"]

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

def degress_of_freedom(attribute):
    return len(valid_values[attribute]) - 1

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
    

def size(data):
    return len(data.index)

def filter_data(attribute, value):
    return examples[examples[attribute] == value]
    
    
def all_one_value(examples, attribute, attribute_value):
    #test that all examples have the same value, examples are a Pandas Dataframe
    filtered_examples = examples[examples[attribute] == attribute_value]
    return len(filtered_examples.index) == len(examples.index)

def possible_range_of_values(attribute):
    return valid_values[attribute]

def extract_most_common_value(examples, attribute):
    #print("most common for ", attribute)
    values = possible_range_of_values(attribute)
    most_common = ""
    highest_count = 0
    for value in values:
        filtered = examples[examples[attribute] == value]
        if len(filtered.index) > highest_count:
            highest_count = len(filtered.index)
            most_common = value  
    return most_common


def chi_square_calculation(examples, target_attribute,attribute_v):
    chi_square = 0.0
    total = len(examples.index)
    if total == 0:
        return 0.0
    positive_filtered_examples = examples[examples[target_attribute] == positive_value]
    negative_filtered_examples = examples[examples[target_attribute] == negative_value]
    
    observed_positive = len(positive_filtered_examples.index)
    expected_positive = (total/2.0)
                
    observed_negative = len(negative_filtered_examples.index)
    expected_negative = (total/2.0)
    
    pos = ((observed_positive - expected_positive)**2)/expected_positive
    neg = ((observed_negative - expected_negative)**2)/expected_negative
    return pos + neg
          
    
def chi_square_test(examples, target_attribute,attribute):
    chi_square_sum = 0.0
    values = possible_range_of_values(attribute)
    for value in values:
        examples_v = examples[examples[attribute] == value]
        chi_square_v = chi_square_calculation(examples_v,target_attribute,value)
        chi_square_sum = chi_square_sum + chi_square_v
    return chi_square_sum
    

def calculate_entropy(examples,target_attribute):
    entropy = 0.0
    total = len(examples.index)
    positive_filtered_examples = examples[examples[target_attribute] == positive_value]
    negative_filtered_examples = examples[examples[target_attribute] == negative_value]
    
    num_positive = len(positive_filtered_examples.index)
    calc_positive = 0.0
    if num_positive != 0:
        calc_positive = -(num_positive/total)*math.log((num_positive/total),2)
            
        
    num_negative = len(negative_filtered_examples.index)
    calc_negative = 0.0
    if num_negative != 0:
        calc_negative = - (num_negative/total)*math.log((num_negative/total),2)
        
        
    return calc_positive + calc_negative
   

def calculate_misclassification_error(examples, target_attribute):
    error = 0.0
    total = len(examples.index)
    if total == 0:
        return 1.0
    positive_filtered_examples = examples[examples[target_attribute] == positive_value]
    negative_filtered_examples = examples[examples[target_attribute] == negative_value]
    
    num_positive = len(positive_filtered_examples.index)
    p_positive = (num_positive/total)
                
    num_negative = len(negative_filtered_examples.index)
    p_negative = (num_negative/total)
        
        
    
    error = 1.0 - max([p_positive,p_negative])
    return error
    
    
def determine_best_attribute_entropy(examples, target_attribute, attributes):
    best_attribute = ""
    max_information_gain = 0.0
   
    #find the entropy S
    entropy_s = calculate_entropy(examples,target_attribute)
    size_of_s = len(examples.index)
    
    
    #loop through attributes and calculate the gain
    for attribute in attributes:
        values = possible_range_of_values(attribute)
        information_gain = entropy_s
        for value in values:
            examples_v = examples[examples[attribute] == value]
            entropy_v = calculate_entropy(examples_v,target_attribute)
            size_of_examples_v = len(examples_v.index)
            weight = (size_of_examples_v/size_of_s)
            information_gain = information_gain - weight*entropy_v
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute
        
       
    return best_attribute

def determine_best_attribute_misclassification_error(examples, target_attribute, attributes):
    best_attribute = ""
    max_information_gain = 0.0
    #find the entropy S
    error_s = calculate_misclassification_error(examples,target_attribute)
    size_of_s = len(examples.index)
    
    #loop through attributes and calculate the gain
    for attribute in attributes:
        values = possible_range_of_values(attribute)
        information_gain = error_s
        for value in values:
            examples_v = examples[examples[attribute] == value]
            error_v = calculate_misclassification_error(examples_v,target_attribute)
            size_of_examples_v = len(examples_v.index)
            weight = (size_of_examples_v/size_of_s)
            information_gain = information_gain - weight*error_v
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute
       
    return best_attribute

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
    if evaluation_criteria == 'entropy':
        a = determine_best_attribute_entropy(examples, target_attribute, attribute_list)
    else:
        a = determine_best_attribute_misclassification_error(examples, target_attribute, attribute_list)
    chi = chi_square_test(examples,target_attribute,a)
    if chi < chi_square_value_for_attribute(a, alpha):
        print("prune: ", a)
        leaf = Leaf()
        leaf.label = extract_most_common_value(examples,target_attribute)
        return leaf
    root.decision_attribute = a
   
    values = possible_range_of_values(a)
    for value in values:
        root.branches[value] = ""
        examples_v = examples[examples[a] == value]
        size_of_examples_v = len(examples_v.index)
        if size_of_examples_v == 0:
            leaf = Leaf()
            leaf.label = extract_most_common_value(examples,target_attribute)
            root.add_branch(value,leaf)
        else:
            if a in attribute_list:
                attribute_list.remove(a)
            node = id3(examples_v,target_attribute,attribute_list, alpha)
            root.add_branch(value, node)
    return root


def split_child(attribute_values, node):
    return node.get_branches()[attribute_values[node.decision_attribute]]

def classify(attribute_values, node):
    if type(node) is Leaf:
        return node.label
    else:
        return classify(attribute_values, split_child(attribute_values,node))
    

def calculate_accuracy(testing_data, id3_root, print_statement):
    count_correct = 0
    count_false = 0
    accuracy = 0
    for i,r in testing_data.iterrows():
        if r["label"] == classify(r,id3_root):
            count_correct = count_correct + 1
        else:
            count_false = count_false + 1
    if count_false == 0:
        accuracy = 100.0
    else:
        accuracy = (count_correct)/(count_correct + count_false)
    print(print_statement, accuracy)
    return accuracy
    

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


evaluation_criteria = "entropy"
alpha = '0.01'
print_statement = "Accuracy (using entropy and alpha value of 0.01): "
id3_entropy_alpha_01 = id3(training,"label",get_attributes(),alpha)
id3_entropy_alpha_01_accuracy= calculate_accuracy(testing_data,id3_entropy_alpha_01, print_statement)

evaluation_criteria = "entropy"
alpha = '0.05'
print_statement = "Accuracy (using entropy and alpha value of 0.05): "
id3_entropy_alpha_05 = id3(training,"label",get_attributes(),alpha)
id3_entropy_alpha_05_accuracy = calculate_accuracy(testing_data,id3_entropy_alpha_05, print_statement)

evaluation_criteria = "entropy"
alpha = '0.5'
print_statement = "Accuracy (using entropy and alpha value of 0.5):  "
id3_entropy_alpha_5 = id3(training,"label",get_attributes(),alpha)
id3_entropy_alpha_5_accuracy = calculate_accuracy(testing_data,id3_entropy_alpha_5, print_statement)

evaluation_criteria = "entropy"
alpha = '0.0'
print_statement = "Accuracy (using entropy and alpha value of 0.0):  "
id3_entropy = id3(training,"label",get_attributes(),alpha)
id3_entropy_accuracy = calculate_accuracy(testing_data,id3_entropy, print_statement)

print("")

evaluation_criteria = "misclassification"
alpha = '0.01'
id3_root = id3(training,"label",get_attributes(),alpha)
calculate_accuracy(testing_data,id3_root, "Accuracy (using misclassification and alpha value of 0.01): ")

evaluation_criteria = "misclassification"
alpha = '0.05'
id3_root = id3(training,"label",get_attributes(),alpha)
calculate_accuracy(testing_data,id3_root, "Accuracy (using misclassification and alpha value of 0.05): ")

evaluation_criteria = "misclassification"
alpha = '0.5'
id3_root = id3(training,"label",get_attributes(),alpha)
calculate_accuracy(testing_data,id3_root, "Accuracy (using misclassification and alpha value of 0.5):  ")

evaluation_criteria = "misclassification"
alpha = '0.0'
id3_root = id3(training,"label",get_attributes(),alpha)
calculate_accuracy(testing_data,id3_root, "Accuracy (using misclassification and alpha value of 0.0):  ")

print("")



    
validation_data = pd.read_table("data/validation.txt", sep=",", 
                                names=["label","cap-shape","cap-surface","cap-color","bruises","oder",
                                        "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
                                        "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
                                        "stalk-color-above-ring","stalk-color-below-ring","veil-type",
                                        "veil-color","ring-number","ring-type","spore-print-color",
                                        "population","habitat"])

validation_data

file = open("validation-best-accuracy.txt", "w")
for i,r in validation_data.iterrows():
    file.write(classify(r,id3_entropy) + "\n")

file.close()
    
    