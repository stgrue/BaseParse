from nltk import Tree
from collections import defaultdict
from nltk.corpus.reader import CategorizedBracketParseCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from nltk.grammar import ProbabilisticProduction
from nltk.grammar import Nonterminal

from utils import simplify

def count_node(node, counter):
    '''Takes a node in a tree and adds 1 to the count of the corresponding
       production (if tree is not a leaf)'''
    if not isinstance(node, Tree) or len(node) == 0:
        return
    if len(node) > 2:
        msg = "Rule '" + str(node.label()) + " -> " + str(node[:]) + " has too many children!"
        raise Exception(msg)
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
                production = ProbabilisticProduction(Nonterminal(lhs), tuple(Nonterminal(sym) for sym in rhs),  prob=counter[lhs][rhs]/lhs_occ)
                f.write(str(production) + "\n")

    with open("root-probs.pcfg", "w+") as g:
        num_sents  = len(ptb_train.parsed_sents())
        for sym in start_sym:
            start_sym[sym] = start_sym[sym] / num_sents
            g.write(sym + "\t" + str(start_sym[sym]) + "\n")
    
if __name__ == "__main__":
    main()
