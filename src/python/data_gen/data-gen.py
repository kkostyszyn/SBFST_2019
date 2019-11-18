#
# A script to generate train/dev/test set
#
import pynini
import functools
import numpy as np
import random

A = functools.partial(pynini.acceptor)
T = functools.partial(pynini.transducer)
e = pynini.epsilon_machine()
zero = e-e
zero.optimize()

# Defining alphabet

alpha = "abcd"
sigma = zero
for x in list(alpha):
    sigma = A(x) | sigma
sigma.optimize()
sigmaStar = (sigma.star).optimize()

#defining edit distance transducer

edits = zero
for x in list(alpha): edits = T(x,"") | edits  # deletion
for x in list(alpha): edits = T("",x) | edits  # insertion
for x in list(alpha):
    for y in list(alpha):
        if x != y:
            edits = T(x,y) | edits                # substitution
edits.optimize()

editExactly1 = sigmaStar + edits + sigmaStar
editExactly1.optimize()


# Utility function that outputs all strings of an fsa

def list_string_set(acceptor):
    my_list = []
    paths = acceptor.paths()
    for s in paths.ostrings():
        my_list.append(s)
    my_list.sort(key=len)
    return my_list

################
# functions for determining the border and generating adversial pairs 
################

def border(fsa,n):
    cofsa = pynini.difference(sigmaStar,fsa)
    cofsa.optimize()
    bpairs = fsa @ editExactly1 @ cofsa     # this is the key insight which gives entire border
    bpairs.optimize()
    sigmaN = pynini.closure(sigma,n,n)
    sigmaN.optimize()
    bpairsN = sigmaN @ bpairs               # here we limit the border to input words of length=n
    bpairsN.optimize()
    return bpairsN
    
    
def build (border, lang, n):
    f = [open("data/adv_data_100k.txt", "w+"),
        open("data/adv_data_10k.txt", "w+"),
        open("data/adv_data_1k.txt", "w+")]
    
    count = 0
    
    for i in range(10):
        by_len(border(lang, n*i), f, count)
            
    print("Total:", count)
    
    for i in range(3):
        f[i].close()
        
    return count
    
def by_len(ex, f, count):
    random_examples=pynini.randgen(ex,10000)
    ps = random_examples.paths(input_token_type="utf8", output_token_type="utf8")

    while not ps.done():
        if ps.istring() and ps.ostring():
            f[0].write(ps.istring() + "\tTRUE\n")
            f[0].write(ps.ostring() + "\tFALSE\n")
            if count % 10 ==0:
                f[1].write(ps.istring() + "\tTRUE\n")
                f[1].write(ps.ostring() + "\tFALSE\n")
                if count %100 ==0:
                    f[2].write(ps.istring() + "\tTRUE\n")
                    f[2].write(ps.ostring() + "\tFALSE\n")
        ps.next()
        count=count+1      


# Utility function that gets the strings of an fsa
# with length from min_len to max_len

def get_pos_string(fsa, min_len, max_len):
    fsa_dict = {}
    for i in range(min_len, max_len + 1):
        fsa_dict[i] = pynini.intersect(fsa, pynini.closure(sigma, i, i))
        # print(list_string_set(fsa_dict[i]))
    return fsa_dict


# Utility function that gets the strings of the complement
# of an fsa with length from min_len to max_len

def get_neg_string(fsa, min_len, max_len):
    fsa_dict = {}
    for i in range(min_len, max_len + 1):
        fsa_dict[i] = pynini.difference(pynini.closure(sigma, i, i), fsa)
        # print(list_string_set(fsa_dict[i]))
    return fsa_dict


# Create {n} random strings from fsa.
# No duplicates in the results.
# The output fsa is the different between the original
# fsa and the delta fsa used to generate unique strings.


def rand_gen_no_duplicate(acceptor, n):
    loop = 10
    for i in range(loop):
        num = int(n + n*i*0.1)
        temp = pynini.randgen(acceptor, npath=num, seed=0, select="uniform", max_length=2147483647, weighted=False)
        rand_list = list_string_set(temp)
        rand_list = list(set(rand_list))
        uniq_len = len(rand_list)
        if uniq_len < n and i < loop - 1:
            print('insufficient random strings')
            continue
        else:
            random.shuffle(rand_list)
            rand_list = rand_list[:n]
            rand_list.sort()
            acceptor = pynini.difference(acceptor, temp)
            return acceptor, rand_list


# Create {num} positive and negative examples from fsa.
# No duplicates in the dataset.


def create_data_no_duplicate(filename, pos_dict, neg_dict, min_len, max_len, num):
    with open(filename, "w+") as f:
        for i in range(min_len, max_len + 1):
            acceptor, results = rand_gen_no_duplicate(pos_dict[i], num)
            pos_dict[i] = acceptor
            for ele in results:
                f.write(ele + "\t" + "TRUE\n")
            acceptor, results = rand_gen_no_duplicate(neg_dict[i], num)
            neg_dict[i] = acceptor
            for ele in results:
                f.write(ele + "\t" + "FALSE\n")
    return pos_dict, neg_dict


# create {num} random strings of positive/negative examples.
# This may be duplicates.


def create_data_with_duplicate(filename, pos_dict, neg_dict, min_len, max_len, num, get_difference):
    with open(filename, "w+") as f:
        for i in range(min_len, max_len + 1):
            pos_fsa = \
                pynini.randgen(pos_dict[i], npath=num, seed=0, select="uniform", max_length=2147483647, weighted=False)
            if get_difference == 1:
                pos_dict[i] = pynini.difference(pos_dict[i], pos_fsa)
            for ele in list_string_set(pos_fsa):
                f.write(ele + "\t" + "TRUE\n")
            neg_fsa = \
                pynini.randgen(neg_dict[i], npath=num, seed=0, select="uniform", max_length=2147483647, weighted=False)
            if get_difference == 1:
                neg_dict[i] = pynini.difference(neg_dict[i], neg_fsa)
            for ele in list_string_set(neg_fsa):
                f.write(ele + "\t" + "FALSE\n")
    return pos_dict, neg_dict


# function that will take the 100k word list and create 10k and 1k from that list
def prune(f, name):
    
    data = open(name).readlines()
    
    small = [open("data/10k/" + f, "w+"),
                open("data/1k/" + f, "w+")]
    
    tr = []
    fl = [] 
    count = 0 
    
    for line in data:
        if count % 2 == 0:
            tr.append(line)
        else:
            fl.append(line)
        count += 1
    count = 0
    
    for x in range(len(tr)):
        if count % 10 == 0:
            small[0].write(tr[x] + fl[x])
            if count %100 ==0:
                small[1].write(tr[x] + fl[x])
        count=count+1   
    
    return True
    
###########################################################
## util functions to generate data and confirm file size ##
###########################################################

def check(n):
    #check 1k, 10k, and 100k to see if all files of type n are of correct length 
    lengths = ["1k", "10k", "100k"]
    #first 1k
    
    for x in lengths:
        test = [open("data/" + x + "/"+n+"_Dev.txt").readlines(),
                open("data/" + x + "/"+n+"_Training.txt").readlines(),
                open("data/" + x + "/"+n+"_Test1.txt").readlines(),
                open("data/" + x + "/"+n+"_Test2.txt").readlines(),
                open("data/" + x + "/"+n+"_Test3.txt").readlines()]
       
        if x == "1k":
            k = 1000
        elif x == "10k":
            k = 10000
        else:
            k = 100000
            
        if len(test[0]) < k:
            print(x + "/" + n + "_Dev incomplete")
        if len(test[1]) < k: 
            print(x + "/" + n + "_Training incomplete")
        if len(test[2]) < k:
            print(x + "/" + n + "_Test1 incomplete")
        if len(test[3]) < k:
            print(x + "/" + n + "_Test2 incomplete")
        if len(test[4]) < k:
            print(x + "/" + n + "_Test3 incomplete")
    
    return True

def check_all():
    tags = open(+"tags.txt").readlines()
    
    for x in tags:
        check(x[:-1])
    return True

def construct_data(n):
    return True
    
def construct_all():
    tags = open("tags.txt").readlines()
    
    for x in tags:
        construct_data(x[:-1])
    return True


############################################
####main body (until functions finished#####
############################################

#my_fsa = A("a").closure() | A("b").closure() | A("c").closure()
path_to_fsa = "/home/student/Desktop/SBFST_2019/"
tags = open(path_to_fsa+"tags.txt")
tags = tags.readlines()

# define hyper-parameters
for x in tags:
    my_fsa = pynini.Fst.read(path_to_fsa + "lib/lib_fst/" + x[:-1] + ".fst")
    x = x[:-1]
    ss_min_len = 10
    ss_max_len = 19
    train_pos_num = 10000
    #dev1_pos_num = 1000
    dev_pos_num = 10000
    
    test1_pos_num = 10000
    test2_pos_num = 5000
    #test3_pos_num = 5000
    #test4_pos_num = 25
    
    ls_min_len = 31
    ls_max_len = 50


    dir_name = "data/100k/" + x

    # generate short strings and construct a dictionary where
    # key=length, value=a list of strings generated by fsa
    if x == "SL.4.2.0":
        
        #FIRST - set up dictionary 
        pos_dict = get_pos_string(my_fsa, ss_min_len, ss_max_len)
        neg_dict = get_neg_string(my_fsa, ss_min_len, ss_max_len)

    
        # create training data with duplicates
        # like the borders - start with a file that has 100k words. from there, prune to have 10k and 1k from that file. 
        pos_dict, neg_dict = \
        create_data_with_duplicate(dir_name + "_Training.txt", pos_dict, neg_dict, ss_min_len, ss_max_len, train_pos_num, 1)
        prune(x + "_Training.txt", dir_name + "_Training.txt")
    
        # create dev and test_1 (no duplicates, no overlap in train/dev/test data)
        pos_dict, neg_dict = create_data_no_duplicate(dir_name + "_Dev.txt", pos_dict, neg_dict, ss_min_len, ss_max_len, dev_pos_num)
        prune(x + "_Dev.txt", dir_name + "_Dev.txt")
        
        pos_dict, neg_dict = create_data_no_duplicate(dir_name + "_Test1.txt", pos_dict, neg_dict, ss_min_len, ss_max_len, test1_pos_num)
        prune(x + "_Test1.txt", dir_name + "_Test1.txt")
 

        # generate long strings
        pos_dict = get_pos_string(my_fsa, ls_min_len, ls_max_len)
        neg_dict = get_neg_string(my_fsa, ls_min_len, ls_max_len)

        # create test_2 (no duplicates)
        create_data_no_duplicate(dir_name + "_Test2.txt", pos_dict, neg_dict, ls_min_len, ls_max_len, test2_pos_num)
        prune(x + "_Test2.txt", dir_name + "_Test2.txt")
        

        # create test_3 (adversarial examples)
        #replace with the border pairs script 
        #create_adversarial_data(dir_name + "_Test3.txt", pos_dict, neg_dict, ls_min_len, ls_max_len, test4_pos_num)
        #c = build(border, my_fsa, 5)

        #print(x)
        
print("Finished!")
