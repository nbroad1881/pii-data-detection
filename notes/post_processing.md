It would be good to check for predictions of I-NAME_STUDENT with no corresponding B-NAME_STUDENT. This means either the token earlier should be B or this token should be B.


### Repeated Names

I noticed that during training that the model would often miss names that were repeated later in the essay.
This will add any name that is repeated and add that as a prediction.

(See documents 9405, 7779)

The name must be capitalized and longer than 1 character to be added.

Potentially has room for improvement if considering prediction scores.



### Combining predictions

#### Stride

When striding over the text, it is good to have overlap. There are a few ways to combine these. 

1. Ignore edges
  - 512 context, 256 stride, only use second 256
2. Average overlap
  - Smooths out predictions
3. Max overlap
  - Might help with recall, hurt precision


#### Character to Token

Probably should revisit the Feedback Prize with span predictions.