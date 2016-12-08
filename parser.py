"""
Python II - Sommersemester 2014
Parsing III - Probabilistisches Parsing
"""
import sys

from nltk import Tree
from collections import defaultdict

from nltk.corpus.reader import CategorizedBracketParseCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from nltk.grammar import Nonterminal
from nltk.grammar import ProbabilisticProduction

from utils import str_flattened

class Prob_CYK_Parser():
    def __init__(self, rules_file, start_file):
        self.rules = defaultdict(lambda: defaultdict(list))
        self.start_prob = defaultdict(float)
        with open(rules_file) as f:
            for line in f:
                lhs, rest = (x.strip() for x in line.split(" -> "))
                rhs = tuple(rest.split()[:-1])
                if len(rhs) == 1:
                    rhs = (rhs[0], "")
                elif len(rhs) > 2:
                    raise Exception("Non-Chomskian!")
                prblt = float(rest.split()[-1][1:-1])
                self.rules[rhs[0]][rhs[1]].append( ProbabilisticProduction(lhs, rhs, prob=prblt) )
        with open(start_file) as g:
            for line in g:
                raw = line.split("\t")
                self.start_prob[raw[0]] = float(raw[1].strip())

    def parse_cyk(self, tokens): # As Argument: List with (word, POS tag)
        '''Probabilistically parse a given sentence (with POS tags given)'''
        self.build_matrix(tokens)

        # Fill in the chart
        for x in range(1, len(tokens)):
            i = 0
            j = x
            while j < len(tokens):
                max_prob = defaultdict(float) # maximum probabilities for each category symbol
                for y in range(0, x):
                    for child1 in (s for s in self.matrix[i][i+y] if s in self.rules):
                        for child2 in (t for t in self.matrix[i+y+1][j] if t in self.rules[child1]):
                            for rule in self.rules[child1][child2]:
                                if rule.prob() * self.matrix[i][i+y][rule.rhs()[0]].inner_prob * self.matrix[i+y+1][j][rule.rhs()[1]].inner_prob > max_prob[rule.lhs()]:
                                    max_prob[rule.lhs()] = rule.prob() * self.matrix[i][i+y][rule.rhs()[0]].inner_prob * self.matrix[i+y+1][j][rule.rhs()[1]].inner_prob
                                    self.matrix[i][j][rule.lhs()] = Entry(rule.lhs(), self.matrix[i][i+y][rule.rhs()[0]], self.matrix[i+y+1][j][rule.rhs()[1]], max_prob[rule.lhs()])
                                  
                i += 1
                j += 1          

        # Find the best root symbol and return the parse tree
        max_prob = 0
        best_root = None
        for root_sym in self.matrix[0][len(tokens)-1]:
            root_prob = self.matrix[0][len(tokens)-1][root_sym].inner_prob * self.start_prob[root_sym]
            if root_prob > max_prob:
                max_prob = root_prob
                best_root = root_sym
        if max_prob > 0:
            return self.matrix[0][len(tokens)-1][best_root].get_tree()
        else:
            return Tree('S', [Tree(tag, [word]) for (word,tag) in tokens])

    def build_matrix(self, tokens):
        '''Build the chart with POS tags at base level'''
        self.matrix = dict()
        for x in range(len(tokens)): 
            self.matrix[x] = dict()
            for y in range(x, len(tokens)):
                self.matrix[x][y] = dict()
        for x in range(len(tokens)):
            (word, tag) = tokens[x]
            self.matrix[x][x][tag] = BaseEntry(word, tag)
        self.complete_unary_rules(tokens, 0)

    def complete_unary_rules(self, tokens, diag):
        '''Go over one diagonal of the chart and fill in possible unary productions'''
        i = 0
        j = diag
        while j < len(tokens):
            modified = True
            while modified == True:
                modified = False
                for child in [c for c in self.matrix[i][j]]:
                    for rule in self.rules[child][""]:
                        if (not rule.lhs() in self.matrix[i][j]) or (rule.prob() * self.matrix[i][j][child].inner_prob > self.matrix[i][j][rule.lhs()].inner_prob):
                            self.matrix[i][j][rule.lhs()] = UnaryEntry(rule.lhs(), self.matrix[i][j][child], rule.prob() * self.matrix[i][j][child].inner_prob)
                            modified = True
                    
            i += 1
            j += 1


class Entry():
    '''Chart entry with two back-pointers'''
    def __init__(self, symbol, left_child, right_child, inner_prob):
        self.symbol = symbol
        self.left_child = left_child
        self.right_child = right_child
        self.inner_prob = inner_prob
    def get_tree(self):
        return Tree(self.symbol, [self.left_child.get_tree(), self.right_child.get_tree()])



class UnaryEntry():
    '''Chart entry with one back-pointer (unary productions)'''
    def __init__(self, symbol, child, inner_prob):
        self.symbol = symbol
        self.child = child
        self.inner_prob = inner_prob
    def get_tree(self):
        return Tree(self.symbol, [self.child.get_tree()])



class BaseEntry():
    '''Chart entry at the pre-terminal level (POS -> word)'''
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag
        self.inner_prob = 1
    def get_tree(self):
        return Tree(self.tag, [self.word])



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 parser.py start end")
        print("(start and end refer to sentence numbers in PTB section 23)")
        sys.exit(1)
    
    ptb_test = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/23/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')
    parser = Prob_CYK_Parser("all-rules.pcfg", "root-probs.pcfg")

    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    with open(str(begin) + "-" + str(end) + ".parse", 'w+') as f:
        for i in range(begin, end+1):
            l = ptb_test.parsed_sents()[i].pos()
            t = parser.parse_cyk(l)
            t.un_chomsky_normal_form()
            f.write(str_flattened(t) + "\n")
