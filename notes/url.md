
There is one document with a "I-URL_PERSONAL" (3202). There are likely not many others in the private set, but it would be good to catch these. It looks like a space accidentally found its way into the url so Spacy split it up.

Perhaps generated data should also have this so that the model can catch it.


## Ideas

1. NER in context
2. Isolate single URLs, predict 1/0 for each. Give a bit of context
  - For urls not in linkedin, facebook, or youtube, a model should easily tell if it is a personal site or not. It shouldn't even need context.
3. Rules based on text (facebook, youtube, linkedin, etc.)
  - No .gov, .edu (not personal)
  - no coursera, wikipedia
  - must have http, ://, www.
  - Maybe include all youtube videos to be safe?
4. Have no context predictions of urls
  - With a large dataset of fake urls, this should be more signal
5. Generate more fake urls
  - easy fake ones (wikipedia, ncbi, etc.)
  - first-last name.com
  - linkedin blogs (not personal)
6. Ask LLM, could "soto" be a first or last name? (https://soto.com/listregister.asp)
  - http://jacobs-fisher.com/listpost.html
  - https://www.jackson.com/list/explorehomepage.htm 


Phrasing for prompting a model
  - Mention your prototype website
  - Mention your blog, video
  - Available online at
  - New improved website
  - new portfolio
  - new content
  - shared stories
  - Our video can be seen here
  - Our team made this video
  - Here is our site