from nltk import Tree
from collections import defaultdict
from nltk.corpus.reader import CategorizedBracketParseCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from utils import simplify

class NonBinaryException(Exception):
    pass

def count_node(node, counter):
    '''Takes a node in a tree and adds 1 to the count of the corresponding
       production (if tree is not a leaf)'''
    if not isinstance(node, Tree) or len(node) == 0:
        return
    elif len(node) == 1 or len(node) > 2:
        msg = "Rule '" + str(node.label()) + " -> " + str(node[:]) + " is not binary!"
        raise NonBinaryException(msg)
    else:
        lhs = node.label()
        rhs = tuple(child.label() for child in node)
        counter[lhs][rhs] += 1

def count_tree(root, counter):
    '''Count all the productions in a tree'''
    count_node(root, counter)
    for child in root:
        count_tree(child, counter)

def main():
    ptb_train = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/((0[2-9])|(1\d)|(2[0-1]))/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')
    ptb_test = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/23/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')

    counter = defaultdict(lambda: defaultdict(int))
    start_sym = defaultdict(float)

    for tree in ptb_train.parsed_sents():
        simplify(tree)
        count_tree(tree, counter)
        start_sym[tree.label()] += 1

    with open("all-rules.pcfg", "w+") as f:
        for lhs in counter:
            lhs_occ = sum(counter[lhs][rhs] for rhs in counter[lhs])
            for rhs in counter[lhs]:
                if len(rhs) != 2:
                    msg = "Rule '" + str(lhs) + " -> " + str(rhs) + " is not binary!"
                    raise NonBinaryException(msg)
                f.write(str(lhs) + "\t" + str(rhs[0]) + "\t" + str(rhs[1]) + "\t" + str(counter[lhs][rhs] / lhs_occ) + "\n")

    with open("root-probs.pcfg", "w+") as g:
        num_sents  = len(ptb_train.parsed_sents())
        for sym in start_sym:
            start_sym[sym] = start_sym[sym] / num_sents
            g.write(sym + "\t" + str(start_sym[sym]) + "\n")
    
if __name__ == "__main__":
    main()
