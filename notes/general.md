

Large number of No-PII data which may have useful data such as urls, names, urls that aren't considered PII. 

It would be good to predict on those to see the hard examples, though early tests suggest False Positives aren't a concern.

Contains PII/No-PII
945/5862

Train with all No-PII?


## Modeling techniques

CRF (flair?)
Packed Levitated (SpanMarker)
MRC?
Layer drop
Multisample dropout
Neftune
AWP
EMA

Assertion model
  - Is "Nicholas Broad" the full name?
  - Instead of predictions at according to the transformer's tokenizer, now it can be 1/0 at spacy tokenizer
  - Check for capitalized terms nearby for names

Tree-based approach for second-level predictions?
  - mean, min, max, etc. for each token
  - text features
  - nearby token features


## Post-processing

Model kept predicting "�" to be "I-USERNAME" (See doc 12106).
  - Make rule that username needs a-zA-Z0-9
  - Since there aren't any "I-USERNAME" labels in the training set, maybe just never predict it!

Never predict
  - "I-USERNAME"
  - "I-EMAIL"
  - "I-URL_PERSONAL"


## Assertion LLM

Names are the most important, so there could be a two-step process for utilizing an LLM to find names.
