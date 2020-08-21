import sys
import re

from numpy import prod
from nltk.corpus import wordnet as wn
def wup_measure(a, b, similarity_threshold):
#a is a word and b is a list
    def get_semantic_field(a):
        return wn.synsets(a)

    if a == b: return 1.0

    interp_a = get_semantic_field(a) 
    interp_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0.0
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score = global_max * interp_weight
    return final_score 

def word_list_wup(a,list_b,thresh):
    _max=0.0
    for b in list_b:
        cur=wup_measure(a,b,thresh)
        if cur>_max:
            _max=cur
    return _max

def answer_wup(sent_a,sent_b,thresh):
    list_a=sent_a.split(' ')
    list_b=sent_b.split(' ')
    score_a=1.0
    for a in list_a:
        score_a=score_a*word_list_wup(a,list_b,thresh)
    score_b=1.0
    for b in list_b:
        score_b=score_b*word_list_wup(b,list_a,thresh)
    if score_a>score_b:
        return score_b
    else:
        return score_a
