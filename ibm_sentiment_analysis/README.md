# ibm_sentiment_analysis

This repo contains code for running [IBM Watson's Tone Analyzer](https://www.ibm.com/cloud/watson-tone-analyzer) to perform sentiment analysis. There are currently 2 versions of the Tone Analyzer, a 2017 version and a 2016 version. The 2017 version includes tones such as anger, fear, joy, sadness, analytical, confident, and tentative, while the 2016 version has a larger range of tones ([anger, disgust, fear, joy, and sadness; analytical, confident, tentative; openness_big5, conscientiousness_big5, extraversion_big5, agreeableness_big5, emotional_range_big5]). Using this tone analyzer requires a free IBMCloud account. For more details on documentation of the Tone Analyzer, visit this site: [https://cloud.ibm.com/apidocs/tone-analyzer](https://cloud.ibm.com/apidocs/tone-analyzer).

## Files Included

* ibm_sentiment_analysis.py
  * File used to run the IBM Tone Analyzer on JSON file containing text.
  * Input file must be formatted as such:
    "text" : <all the text to be analyzed>
    See data/few_sent.json for example.
  * Some changes need to be made in the file to use the API
* ibm_results
  * Results of the analysis
  * Currently contains few_sent_res.json, the result of running few_sent.json (in data).
  * Contain a document score and individual sentence scores, which are confidence levels of each tone for each sentence. See documentation for more details.

## Running Code

Before running the code, there are a few additional steps that need to be taken.
1. Create a free IBMCloud account to access the Tone Analyzer. Go to [this website](https://www.ibm.com/cloud/watson-tone-analyzer?p1=Search&p4=43700050290119172&p5=e&cm_mmc=Search_Google-_-1S_1S-_-WW_NA-_-ibm%20tone%20analyzer_e&cm_mmca7=71700000061102158&cm_mmca8=aud-382859943522:kwd-567122059112&cm_mmca9=EAIaIQobChMI9fSF7Ji36wIVxP7jBx3lpwfvEAAYASAAEgK4cfD_BwE&cm_mmca10=405936285068&cm_mmca11=e&gclid=EAIaIQobChMI9fSF7Ji36wIVxP7jBx3lpwfvEAAYASAAEgK4cfD_BwE&gclsrc=aw.ds) and click "Get started free".
2. Access the API key and service URL.
3. In ibm_sentiment_analysis.py, add your API key and service URL (as a string) in the variables at the top of the function.

To use the analyzer, run
```bash
  python3 ibm_sentiment_analysis.py <input_file> <output_file>
```
* ```<input_file>``` is a JSON file of the input sentences to be analyzed
* ```<output_file>``` is a JSON file containing the results

## Limitations

Since the IBMCloud account is free, we are using the Lite version of the Analyzer, which has the following limitations:
* You can submit no more than 128 KB of total input content.
* You can submit no more than 1000 individual sentences.
* The service analyzes the first 1000 sentences for document-level analysis and only the first 100 sentences for sentence-level analysis.  
* 2500 API calls per month with Lite

Additionally, this code only analyzes general tones, but there is a customer-engagement tone analyzer for more specific purposes. See documentation for further details.