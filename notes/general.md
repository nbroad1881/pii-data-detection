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

### March 31

- Looked over [Feedback Prize ESW](https://www.kaggle.com/competitions/feedback-prize-2021/discussion?sort=published) 1st place second stage lgbm [discussion](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177) and [model](https://www.kaggle.com/code/wht1996/feedback-lgb-train) and the [notebook](https://www.kaggle.com/code/chasembowers/sequence-postprocessing-v2-67-lb) it was based off of. Tried to use OOF predictions and got bad results. May try again with in-fold predictions on as much data as possible. Will also switch to binary predictions
- Adding more locales to address and phone numbers (en_GB, es_ES, etc.). Must have latin characters
  - With only 2 addresses in training data and 
- Training d1xl with lora, but it's doing worse than d1b
- d1b got 971 on validation, only 948 on lb
- d3l trained on all data got 968 using piidd lib
- use mistral 7b 0.1/0.2 with 30k position embeddings deleted, 1-2 layers removed so it can be used in fp16 on single gpu
- need different model types (llm, lstm, longformer)
- potentially try something like this where predict 1 B class, and 1 class for each label + outside. [link](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/315887)


### April 2

- Tried combining url classifier, did worse.
  - Maybe overfit to train set ()
- second stage with only token predictions from 3 models gets 936 on competition dataset (some leak)
- explored predictions more, can potentially use oof predictions to help write rules
- different thresholds for each class
- need another validation set besides competition (there aren't enough entities)


### April 3

- Noticed high scores in two different categories for overlap (I-name, O). Maybe consider throwing out if not enough context? e.g. on idx >=1, ignore scores in first 10-20 tokens. Have second model decide?
- No B,I seems like a good idea
- Use str.istitle() as feature before classification to help with names
- there are a few urls like "http://www.tate.com/" which don't have a long path. Maybe adding more of these would help the model understand
- made smaller mistral (30/28 layers) to use in fp16 on t4

### April 4

- Username must be lowercase
  - OOF predictions have high B-USERNAME scores, lowest is 0.6
  - OOF predictions will have high O scores for non-username tokens
- might be issue with labeling \n for street addresses
  - If the tokenizer cannot handle \n then, it cannot predict it!
    - deberta-v3 cannot see \n by default
    - deberta-v2 does not have \n
    - deberta-v1 has Ċ for \n
  - found bug in tokenization! wasn't labeling \n even if token was added!!
- created mixtral-v3 dataset that has some urls without long paths


### April 5

- submissions low score (954) :(
- training without BI
- maybe have LSTM head that uses embeddings + class predictions + edge labels (spacy tokens) to predict start/end
- need smarter second stage
- need reliable CV besides train set
- need to know which "empty" samples should be in train set
- double-check why fold 2 struggles so much