from io import StringIO
import pandas as pd
from datasets import Dataset
import numpy as np
from scipy.special import softmax


from piidd.processing.post import get_all_preds


# for checking phone number processing
test_df = pd.read_csv(StringIO("""document	label	token_text
Doc1	B-PHONE_NUM	12893849239283
Doc1	I-PHONE_NUM	12893849239283
Doc1	B-PHONE_NUM	12893849.239283
Doc2	B-PHONE_NUM	1289x3849239283
Doc2	B-PHONE_NUM	1(289)3849239283
Doc3	B-URL_PERSONAL	https://wikipedia.org
Doc3	O	Another text
Doc4	B-URL_PERSONAL	https://example.gov
Doc4	O	Yet another text
Doc5	B-URL_PERSONAL	https://example.edu"""), sep="\t")


# for checking URL processing
test_df = pd.read_csv(StringIO("""document	label	token_text
Doc1	B-URL_PERSONAL	https://www.example.com
Doc1	I-URL_PERSONAL	www.example.com/page
Doc1	B-URL_PERSONAL	Some text
Doc2	B-URL_PERSONAL	http://coursera.org
Doc2	B-NAME_PERSONAL	Nicholas
Doc3	B-URL_PERSONAL	https://wikipedia.org
Doc3	O	Another text
Doc4	B-URL_PERSONAL	https://example.gov
Doc4	O	Yet another text
Doc5	B-URL_PERSONAL	https://example.edu"""), sep="\t")


# for checking name processing



ds = Dataset.from_parquet("/drive2/kaggle/pii-dd/piidd/inference/outputs/d3l-floral-bird-887-mpware/ds.pq")
tds = Dataset.from_parquet("/drive2/kaggle/pii-dd/piidd/inference/outputs/d3l-floral-bird-887-mpware/tds.pq")
preds = np.load("/drive2/kaggle/pii-dd/piidd/inference/outputs/d3l-floral-bird-887-mpware/preds.npy")

preds = softmax(preds, -1)


id2label = {0: 'B-EMAIL',
 1: 'B-ID_NUM',
 2: 'B-NAME_STUDENT',
 3: 'B-PHONE_NUM',
 4: 'B-STREET_ADDRESS',
 5: 'B-URL_PERSONAL',
 6: 'B-USERNAME',
 7: 'I-ID_NUM',
 8: 'I-NAME_STUDENT',
 9: 'I-PHONE_NUM',
 10: 'I-STREET_ADDRESS',
 11: 'O'}

get_all_preds(preds, ds, tds, id2label, return_all_token_scores=True)