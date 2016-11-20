"""
Python II - Sommersemester 2014
Parsing III - Probabilistisches Parsing
"""
from nltk import Tree
from collections import defaultdict

from nltk.corpus.reader import CategorizedBracketParseCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from utils import get_tagged_words

class Prob_CYK_Parser():
    def __init__(self, filename):
        #self.rules = list()
        self.rules = list()
        with open(filename) as f:
            for line in f:
                raw = line.split("\t")
                self.rules.append(Rule(raw[0], (raw[1], raw[2]), float(raw[3])))

    def parse_cyk(self, tokens): # Als Argumente: Liste mit (Wort, POS-Tag)
        self.build_matrix(tokens)

        # Schritt 2 bis n:
        # Gehe Diagonale entlang und schau, ob Regeln angewandt werden können, fülle dann Zellen aus
        for x in range(1, len(tokens)):
            i = 0
            j = x
            while j < len(tokens):
                max_prob = defaultdict(float) # maximale Wahrscheinlichkeiten für jede Kategorie
                for y in range(0, x):
                    for rule in self.rules:
                        if rule.rhs[0] in self.matrix[i][i+y] and rule.rhs[1] in self.matrix[i+y+1][j]:
                            if rule.prob * self.matrix[i][i+y][rule.rhs[0]].inner_prob * self.matrix[i+y+1][j][rule.rhs[1]].inner_prob > max_prob[rule.lhs]:
                                max_prob[rule.lhs] = rule.prob * self.matrix[i][i+y][rule.rhs[0]].inner_prob * self.matrix[i+y+1][j][rule.rhs[1]].inner_prob
                                self.matrix[i][j][rule.lhs] = Entry(rule.lhs, self.matrix[i][i+y][rule.rhs[0]], self.matrix[i+y+1][j][rule.rhs[1]], max_prob[rule.lhs])
                              
                i += 1
                j += 1          


        if "S" in self.matrix[0][len(tokens)-1]:
            return self.matrix[0][len(tokens)-1]["S"].get_tree()
        else:
            return None

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


    def get_tree(self, symbol, x, y):
        if not self.matrix[x][y]:
            return

        if self.matrix[x][y][symbol][1][1] == -1: # Rekursionsbasis
            return Tree(symbol, [Tree(self.words[y], [])])

        return Tree(symbol, [  self.get_tree(self.matrix[x][y][symbol][1][0], self.matrix[x][y][symbol][1][1], self.matrix[x][y][symbol][1][2])  , self.get_tree(self.matrix[x][y][symbol][2][0], self.matrix[x][y][symbol][2][1], self.matrix[x][y][symbol][2][2])    ])



class Rule():
    def __init__(self, lhs, rhs, prob):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob
    def __str__(self):
        return str(self.lhs) + " -> " + str(self.rhs[0]) + " " + str(self.rhs[1]) + ", " + str(self.prob)



class Entry():
    def __init__(self, symbol, left_child, right_child, inner_prob):
        self.symbol = symbol
        self.left_child = left_child
        self.right_child = right_child
        self.inner_prob = inner_prob
    def get_tree(self):
        return Tree(self.symbol, [self.left_child.get_tree(), self.right_child.get_tree()])



class BaseEntry():
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag
        self.inner_prob = 1
    def get_tree(self):
        return Tree(self.tag, [self.word])



def main():
    ptb_test = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/23/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')
    parser = Prob_CYK_Parser("ctr2.txt")

    #for tree in ptb_test.parsed_sents():

if __name__ == "__main__":
    ptb_test = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/23/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')
    parser = Prob_CYK_Parser("all-rules.pcfg")

    for i in [0]:
        l = get_tagged_words(ptb_test.parsed_sents()[i])
        t = parser.parse_cyk(l)
        print(t)
    
#    parser = Prob_CYK_Parser()
#    
#    test = ["the", "dog", "salespeople", "sold", "the", "dog", "biscuits"]
#    r = parser.recognize_cyk(test, rules, lexicon)
#    print(r)
#    print("Satz in der Sprache?", r[0])
#    tree, prob = parser.parse_cyk(test, rules, lexicon)
#    print(prob, tree)
    
