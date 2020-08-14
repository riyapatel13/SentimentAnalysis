# https://github.com/watson-developer-cloud/python-sdk/blob/master/examples/tone_analyzer_v3.py



import json
import os
from os.path import join
from ibm_watson import ToneAnalyzerV3
from ibm_watson.tone_analyzer_v3 import ToneInput
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Authentication via IAM
authenticator = IAMAuthenticator('ariKWpUe_6Mx5YAQRtAzJ6VV7Bu6qAeKzlVlZSKC9Y0_')
service = ToneAnalyzerV3(
    version='2016-05-19:',
    # include versioning option
    authenticator=authenticator)
service.set_service_url('https://api.us-east.tone-analyzer.watson.cloud.ibm.com/instances/7a24ce0a-afd9-49c3-980e-fe89b178474d')

# Authentication via external config like VCAP_SERVICES
# service = ToneAnalyzerV3(version='2017-09-21')
# service.set_service_url('https://api.us-east.tone-analyzer.watson.cloud.ibm.com/instances/7a24ce0a-afd9-49c3-980e-fe89b178474d')

# input must be json with "text":<whole paragraph>
infile = "test_file.json"
outfile = "ibm_results.json"

# Python File I/O
def read_file(path):
    with open(path, "rt") as f:
        return f.readlines()
def write_file(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def analyze_tone(sentence):
    pass

'''
intext = read_file(infile)
with open(join(os.getcwd(), infile)) as tone_json:
    tone = service.tone(json.load(tone_json)['text'], content_type="text/plain").get_result()
print(json.dumps(tone, indent=2))
'''
with open(join(os.getcwd(),
               infile)) as tone_json:
    tone = service.tone(
        tone_input=json.load(tone_json),
        content_type='application/json').get_result()
#print(json.dumps(tone, indent=2))

write_file(outfile, json.dumps(tone, indent=2))
print("wrote to file")