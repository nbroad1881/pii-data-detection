
There is one document with a "I-URL_PERSONAL" (3202). There are likely not many others in the private set, but it would be good to catch these. It looks like a space accidentally found its way into the url so Spacy split it up.


## Ideas

1. NER in context
2. Isolate single URLs, predict 1/0 for each. Give a bit of context
  - For urls not in linkedin, facebook, or youtube, a model should easily tell if it is a personal site or not. It shouldn't even need context.
3. Rules based on text (facebook, youtube, linkedin, etc.)
  - No .gov, .edu (not personal)
  - no coursera, wikipedia
  - must have http, ://, www.