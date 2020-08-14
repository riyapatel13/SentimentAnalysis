from stat_parser import Parser
import re
import string
import argparse

'''
data_format.py

This file formats the data so that it can be used for sentiment analysis. It
uses a parser using the CKY algorithm from the "Natural Language Processing" 
course by Michael Collins (https://class.coursera.org/nlangp-001/class). The
parser uses the PennTreebankTokenizer that uses regular expressions to tokenize 
text as in Penn Treebank (http://www.cis.upenn.edu/~treebank/tokenizer.sed).

This tokenizer performs the following steps:
    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line
ex: s = "They'll save and invest more."
    s = s.tokenize()
    >>> ['They', "'ll", 'save', 'and', 'invest', 'more', '.'] 

The parser then constructs a grammar tree using NLTK's Tree class (separates the
phrases and tags each phrase with a part-of-speech tag).
ex: s = "This is a sentence."
    s = parser.parse(s)
    >>> (S
        (NP (DT this))
        (VP (VBZ is) (NP (DT a) (NN test) (NN sentence)))
        (. .)
        )
The file first splits up the paragraph into distinct sentences using regex
(https://stackoverflow.com/questions/4576077/python-split-text-on-sentences).
Since there are nested phrases and tags, this file uses a stack 
(https://stackoverflow.com/questions/5454322/python-how-to-match-nested-parentheses-with-regex).
to evaluate the beginning and end of a phrase, and it removes the POS tag, punctuation, 
and  parentheses, and stores each distinct phrase on a new line of a file that it writes to.

This data can then be fed into the sentiment analysis tool.
'''


'''
Uses regex to find grouping of parentheses.
:param line: input string (string version of grammar tree)
:param opendelim: opening character to match
:param closedelim: closing character to match
    https://stackoverflow.com/questions/5454322/python-how-to-match-nested-parentheses-with-regex
:return: None
'''
def matches(line, opendelim='(', closedelim=')'):
    
    stack = []

    for m in re.finditer(r'[{}{}]'.format(opendelim, closedelim), line):
        pos = m.start()

        if line[pos-1] == '\\':
            # skip escape sequence
            continue

        c = line[pos]

        if c == opendelim:
            stack.append(pos+1)

        elif c == closedelim:
            if len(stack) > 0:
                prevpos = stack.pop()
                yield (prevpos, pos, len(stack))
            else:
                # error
                print("encountered extraneous closing quote at pos {}: '{}'".format(pos, line[pos:] ))
                pass

    if len(stack) > 0:
        for pos in stack:
            print("expecting closing quote to match open quote starting at: '{}'"
                  .format(line[pos-1:]))


'''
Creates stack to organize phrases (using parens) and removes POS tag.
:param tree: grammar tree (type = 'nltk.tree.Tree')
:return: stack of phrases
'''
def create_stack(tree):

    stack = []

    for part in str(tree).split():
        if part[0] == "(":
            stack.append("(")
        else:
            count = 0
            word = ""
            for char in part:
                if char != ")":
                    word += char
                else:
                    count += 1
            stack.append(word)
            for paren in range(count):
                stack.append(")")

    return stack

'''
Converts stack contents to string.
:param stack: stack containing phrases created by create_stack
:return: string version of stack
'''
def stack_to_string(stack):

    s = str(stack)
    s = re.sub("\'","", s)

    s = re.sub("\"","", s)
    s = s.replace(",","")
    #s = s.replace(" ", "")
    i = 0
    while i < len(s):
        if s[i] == "[":
            s = s[:i] + "(" + s[i+1:]
        if s[i] == "]":
            if s[i-1].isalnum():
                s = s[:i] + ')' + s[i+1:]
            else:
                s = s[:i] + ')' + s[i+1:]
        i+=1
    
    return s

def strip_space(sent):
    return re.sub(" +", " ", sent)

'''
Creates a list of parsed phrases given a single sentence.
:param sentence: sentence from the data
:return: list of phrases
'''  
def create_phrase_list(sentence):

    #remove ending punctuation
    if sentence[-1] in string.punctuation:
        sentence = sentence[:-1]   
        if sentence == "":
            return
   
    stack = create_stack(data_parser.parse(sentence)) 
    sent = stack_to_string(stack)

    phrase_list = []

    # every phrase
    for openpos, closepos, level in matches(sent):
        # cleaning up string by removing parentheses
        phrase = sent[openpos:closepos]
        phrase = phrase.replace('(','')
        phrase = phrase.replace(')','')
        # remove inner punctuation
        phrase = phrase[:-1]
        phrase = strip_space(phrase.strip())

        if phrase not in phrase_list:
            phrase_list.append(phrase)

    return phrase_list


# Python File I/O
def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

'''
Splits a string into sentences using regex.
:param text: string containing the contents of the data
https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
:return: list of sentences
'''
def split_into_sentences(text):

    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"


    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

#reads file and parses all sentences
'''
Parses entire document using functions above.
:param path: file path of data
:return: list of phrases
'''
def parse_doc(path):

    tos = readFile(path)

    # edit file to make it readable for parser
    # this needs some customization depending on the text format
    tos = tos.replace("\n",". ")
    tos = tos.replace("(", "(")
    tos = tos.replace(")", "")
    tos = tos.replace('"\\"', "")
    tos = tos.replace("/", "")
    tos = tos.replace(":\n", ".")
    tos = tos.replace(":",",")
    
    phrase_list = []
    for sentence in split_into_sentences(tos):
        x = create_phrase_list(sentence)
        if x != None:
            phrase_list += x
    return phrase_list


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description = "Data parser - given an input file (in paragraph form or each sentence in a new line), it will return a new file with the parsed data that can be fed into the sentiment analysis tool")
    argparser.add_argument("input_file",
                        type=str,
                        help="file containing data to be parsed")
    argparser.add_argument("output_file",
                        type=str,
                        help="file containing parsed data to be fed into the sentiment analysis tool")

    args = argparser.parse_args()
    data_parser = Parser()
    
    # intermediary step - writes entire list of phrases to text file - not necessary (used for debugging)
    list_file = (args.output_file).split('.')
    list_file = list_file[0]+"_list."+list_file[1]

    writeFile(list_file ,str(parse_doc(args.input_file)))
    tos = readFile(list_file)

    tos = tos[2:]
    tos = tos[:-2]
    tos = tos.replace("(","")
    tos = tos.replace(")","")
    tos = tos.replace("[","")
    tos = tos.replace("]","")
    tos = tos.replace(" ,", ",")
    # this makes it go into separate the phrases into new lines
    tos = tos.replace("\', \'","\n") 
    writeFile(args.output_file, tos)
    print(f"The parsed data should be located in a file named \"{args.output_file}\". To see all the data in a list format, you can open the file \"{list_file}\"")
