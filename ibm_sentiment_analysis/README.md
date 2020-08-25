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