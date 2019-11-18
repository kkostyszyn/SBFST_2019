# 
#
# The main function is 'border' which takes as input a FSA and a positive integer n
#
# and returns a FST such that (x,y) belongs tothe relation of FST *iff* the following is true:
#    (1) x belongs to the language of the FSA
#    (2) the length of x is n
#    (3) y belongs to the complement of the language of the FSA
#    (4) d(x,y) = 1   ; i.e. the string edit distance between x and y is 1. 
#
#
#   code originally by Jeff Heinz, with modifications 

import pynini
import functools
#from prettytable import PrettyTable

A = functools.partial(pynini.acceptor, token_type="utf8")
T = functools.partial(pynini.transducer, input_token_type="utf8", output_token_type="utf8")
e = pynini.epsilon_machine()
zero = e-e
zero.optimize()

################################
# Defining sigma and sigmastar #
################################

alphabet="abcd"
sigma = zero
for x in list(alphabet): sigma = A(x) | sigma
sigma.optimize()

sigmaStar = (sigma.star).optimize()

        
#######################
# The EDIT Transducer #
#######################

edits = zero
for x in list(alphabet): edits = T(x,"") | edits  # deletion
for x in list(alphabet): edits = T("",x) | edits  # insertion
for x in list(alphabet):
    for y in list(alphabet):
        if x != y:
            edits = T(x,y) | edits                # substitution
edits.optimize()

editExactly1 = sigmaStar + edits + sigmaStar
editExactly1.optimize()

######################
# GETTING THE BORDER #
######################

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


##############
# THAT'S IT! #
##############

######################
# BUILDING THE FILES #
######################

def build (border, lang, n):
    f = [open("adv_data_100k.txt", "w+"),
        open("adv_data_10k.txt", "w+"),
        open("adv_data_1k.txt", "w+")]
    
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
    
    #return True


######################
# RUNNING AN EXAMPLE #
######################

#####################
# EXAMPLE LANGUAGES #
#####################

a = A("a")
b = A("b")
def lg_containing_str(x,i):
    return (sigmaStar + pynini.closure(b,i,i) + sigmaStar).minimize()
def lg_containing_ssq(x,i):
    return (pynini.closure(sigmaStar + x + sigmaStar,i,i)).minimize()

###############
# SL Examples #
###############

sl = dict()
sl[0] = sigmaStar - lg_containing_str(b,2)  # SL2 , forbidden bb
sl[1] = sigmaStar - lg_containing_str(b,4)  # SL4 , forbidden bbbb
sl[2] = sigmaStar - lg_containing_str(b,8)  # SL8 , forbidden bbbbbbbb


###############
# SP Examples #
###############

sp=dict()
sp[0] = sigmaStar - lg_containing_ssq(b,2)     # SP2 , forbidden bb
sp[1] = sigmaStar - lg_containing_ssq(b,4)     # SP4 , forbidden bbbb
sp[2] = sigmaStar - lg_containing_ssq(b,8)     # SP8 , forbidden bbbbbbbb

lt=dict()
# LT2 , at least one bb
lt[0] = lg_containing_str(b,2)

# LT4 , at least one bbbb or at least one aaaa
lt[1] = pynini.union(lg_containing_str(b,4), lg_containing_str(a,4))
# lt[1] = lg_containing_str(b,4) + lg_containing_str(a,4)


# Minimizing the acceptors

pair_names=[(sl,'sl'),
            (sp,'sp'),
            (lt,'lt')]
lg_classes = dict(enumerate(pair_names))

for x in lg_classes:
    lg_class,name = lg_classes[x]
    for i in list(range(len(lg_class))):
        lg_class[i].optimize()


### All previous code has been consolidated into this
### build() and by_len() functions above. 

count = build(border, lt[0], 5)

#right now the only bug i'm seeing is that count isn't returning properly? 
print("total pairs:"+str(count))

