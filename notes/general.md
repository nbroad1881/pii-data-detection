# General thoughts about the competition

Large number of No-PII data which may have useful data such as urls, names, urls that aren't considered PII. 

It would be good to predict on those to see the hard examples, though early tests suggest False Positives aren't a concern.

Contains PII/No-PII
945/5862

Train with all No-PII?


## Annotation Guidelines

[Link](https://storage.googleapis.com/kaggle-forum-message-attachments/2631330/20264/Annotation%20Guidelines%20Kaggle.pdf) provided by hosts.


## Modeling techniques

CRF (flair?)
Packed Levitated (SpanMarker)
MRC?
Layer drop
Multisample dropout
Neftune
AWP
EMA
[Gliner](https://huggingface.co/urchade/gliner_largev2)

## Assertion model
  - Is "Nicholas Broad" the full name?
  - Instead of predictions according to the transformer's tokenizer, now it can be 1/0 at spacy tokenizer
  - Check for capitalized terms nearby for names

## Tree-based approach for second-level predictions?
  - mean, min, max, etc. for each token
  - text features
  - nearby token features

## Span prediction as second stage
  - Average over likely candidates and use average span embeddings for prediction
  - Use mean embeddings of a phrase that describes the class as an additional feature
    - "A name of a person. Not the name of a school, organization, or a famous person."


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



## Public LB

[Thread showing which labels are not in public set](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/480067)

No phone num.
No I-username, I-email.
There definitely are:
  - B/I-Name student
  - B/I-url
  - B-username
  - B-ID
  - B/I-address
  - B-email


## Gameplan

### March 29

- Generating more data doesn't seem to help.
- Second stage is probably the secret to boosting the score further
  - First stage train on mixtral + 2 fold of provided data (validate on 1 fold) 
    - deberta-v3-large
    - deberta-v2-xlarge
    - deberta-large
    - gliner-base (large?)
    - phi-2 + bi-lstm
    - names-base
    - regex
  - Second stage train on 1 fold (validate on other)
    - some lightgbm model
    - lots of hand crafted features
      - in parenthesis
      - near period
      - near \n
      - capitalize
      - length
      - etc.
    - After validation, train on 2 folds
