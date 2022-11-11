from src.transformers.tokenization_utils import PreTrainedTokenizer
from src.transformers.models.bert.tokenization_bert import BertTokenizer

tokenizer = PreTrainedTokenizer()
tokenizer2 = BertTokenizer("vocab.txt")

input_tekst = "Maril on paha tuju, aga see on fine."

#print(tokenizer.tokenize(input_tekst))
#print(tokenizer(input_tekst))
# print(tokenizer.added_tokens_encoder)

print()
print(tokenizer2.tokenize(input_tekst))
print(tokenizer2(input_tekst))