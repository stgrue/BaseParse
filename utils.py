from nltk.tree import Tree
from itertools import chain

def simplify(t):
    '''Pre-process tree to use as base for PCFG'''
    unlex(t)
    strip_labels(t)
    t.chomsky_normal_form()

def strip_labels(t):
    '''Simplifies labels of given tree to contain only phrases'''
    if t.label()[0] != '-':
        new_label = str(t.label()).split("-")[0]
    else:
        new_label = t.label()
    new_label = new_label.split("=")[0]
    t.set_label(new_label)
    for child in t:
        strip_labels(child)
    

def is_leaf(t):
    '''Determine if a given tree is a leaf'''
    return (not isinstance(t, Tree)) or (len(t) == 0)

def unlex(t):
    '''Remove all lexical items from a given tree'''
    t[:] = [child for child in t if not is_leaf(child)]
    for i in range(len(t)):
        unlex(t[i])

def remove_unary(t):
    '''Collapse all unary expansions into bottom-most node'''
    if not isinstance(t, Tree):
        return
    if len(t) == 1:
        t.set_label(t[0].label())
        t[:] = t[0][:] 
    if len(t) == 1:
        remove_unary(t)
    elif len(t) > 1:
        for child in t:
            remove_unary(child)

def str_flattened(t):
    '''Print tree in one line (for use with evalb)'''
    return "( " + " ".join(s.strip() for s in str(t).split('\n')) + ")"
