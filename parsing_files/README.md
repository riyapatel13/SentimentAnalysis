# parsing_files

This repo contains code to parse sentences and create more training data. 

## Files Included

* data_format.py
  * Reads data from input text file (in paragraph form or line-separated sentences) and parses it into separate phrases that can be fed into the sentiment analysis files. The file first splits up the paragraph into distinct sentences using regex. Since there are nested phrases and tags, this file uses stack evaluation to distinguish phrases and removes POS tags and unnecessary punctuation and stores each distinct phrase on a new line of a file that it writes to.
* stat_parser
  * Folder containing files used in data_format.py to tokenize and parse the data into phrases. The parser is created using the CKY algorithm from the ["Natural Language Processing"](https://class.coursera.org/nlangp-001/class) course by Michael Collins. It uses the PennTreebankTokenizer that uses regular expressions to tokenize text as in [Penn Treebank](https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html). See the following link or the documentation in data_format.py for more details.

## Running Code
  
  To format the data so that it can properly be fed into the sentiment analysis tools, run
  ```bash
  python data_format.py <input file> <output file>
  ```
  * ```<input file>``` is the file containing sentences that need to be parsed (either line-separated or in paragraph form)
  * ```<output file>``` contain the results of the parsed data
  * Another file containing the output in a list format should also be created. For more information, run ```python data_format.py -h```.