
import json
import os
from os.path import join
from ibm_watson import ToneAnalyzerV3
from ibm_watson.tone_analyzer_v3 import ToneInput
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import argparse

'''
TODO:
- documentation
- accept JSON file or text file as inputs
- fix new line issue

ibm_sentiment.py

https://github.com/watson-developer-cloud/python-sdk/blob/master/examples/tone_analyzer_v3.py

- Tones:
    - 2017 : [anger, fear, joy, sadness; analytical, confident, tentative]
    - 2016 : [anger, disgust, fear, joy, and sadness; 
              analytical, confident, tentative; 
              openness_big5, conscientiousness_big5, extraversion_big5, agreeableness_big5, emotional_range_big5]
- 2017 version will only report tones with over 0.5 confidence
- You can submit no more than 128 KB of total input content and no more than 1000 individual sentences 
in JSON, plain text, or HTML format. The service analyzes the first 1000 sentences for document-level 
analysis and only the first 100 sentences for sentence-level analysis.
- 2500 API calls per month with Lite
- Currently using general tone, but there is also a customer-engagement version (for customer service)
'''

'''
Insert API key in IAMAuthenticator.
The current version of the Tone Analyzer can be found on the documentation site:
https://cloud.ibm.com/apidocs/tone-analyzer?_ga=2.86587288.1693115604.1593467976-244482230.1591637228&_gac=1.238872244.1591637236.EAIaIQobChMI0vvu2N7y6QIVy9SzCh2jewVbEAAYASAAEgKi-_D_BwE&cm_mc_uid=25906634997715796743523&cm_mc_sid_50200000=91288221593526595253&cm_mc_sid_52640000=58274171593526595279
Currently 2 versions: 2017-09-21, 2016-05-19
The service URL can be found with the API key. 
'''
# WRITE YOUR API KEY AND SERVICE URL HERE 
api_string = 'ariKWpUe_6Mx5YAQRtAzJ6VV7Bu6qAeKzlVlZSKC9Y0_'
service_url_string = 'https://api.us-east.tone-analyzer.watson.cloud.ibm.com/instances/7a24ce0a-afd9-49c3-980e-fe89b178474d'

def init_analyzer(api_string, service_url_string):
    authenticator = IAMAuthenticator(api_string)
    analyzer = ToneAnalyzerV3(
        version='2016-05-19',
        authenticator=authenticator)
    analyzer.set_service_url(service_url_string)
    
    return analyzer

'''
Use service.tone to analyze the tone of the input and json.dumps to format as json. 
tone_input: input text
content_type: [application/json, text/plain, text/html]
'''

'''
This code segment will take the command-line input and analyze the tone of the sentences typed.
'''
'''
sent = input('Hello, please provide a sentence for analysis.\n') 

print(
    json.dumps(
        analyzer.tone(
            tone_input=sent,
            content_type="text/plain").get_result(),
        indent=2))
'''

'''
This code will read a JSON file and search for the 'text' section and analyze all the sentences in the text section.
optional parameter: sentences (bool) -> will return individual sentence analysis. True by default.

To run only on JSON files, make the following changes:
    text = json.load(tone_json)
    tone = analyzer.tone(tone_input=text, content_type="application/json").get_result()
'''
def analyze(analyzer, input_file, output_file):
    with open(input_file, 'r') as tone_json:
        text = json.load(tone_json)['text']
        tone = analyzer.tone(tone_input=text, content_type="text/plain").get_result()

    with open(output_file, 'w') as outfile:
        json.dump(tone, outfile, indent=2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="IBM Tone Analyzer - given an input JSON file with format \"text\": [text to be analyzed], it will return a JSON file with individual sentence tone as well as overall text tone.")
    argparser.add_argument("input_file_path",
                            type=str,
                            help="Path to JSON file to be analyzed. The file must contain \"text\": [input text] name/value pair.")
    argparser.add_argument("output_file_path",
                            type=str,
                            help="File containing tone analysis of text.")
    args = argparser.parse_args()

    service = init_analyzer(api_string, service_url_string)
    analyze(service, args.input_file_path, args.output_file_path)

    print(f"The tone analysis should be located in a file named \"{args.output_file_path}\".")