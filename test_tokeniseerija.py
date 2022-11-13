from src.transformers.tokenization_utils import PreTrainedTokenizer
from src.transformers.models.bert.tokenization_bert import BertTokenizer
from src.transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
from src.transformers.models.bert.configuration_bert import BertConfig
import torch

tokenizer = PreTrainedTokenizer()
tokenizer2 = BertTokenizer("vocab.txt")

input_tekst = "Maril on paha tuju, aga see on fine."

#print(tokenizer.tokenize(input_tekst))
#print(tokenizer(input_tekst))
# print(tokenizer.added_tokens_encoder)

#print()
print(tokenizer2.tokenize(input_tekst))
print(tokenizer2(input_tekst))

config = BertConfig()
embedding = BertEmbeddings(config)
#model = BertModel(config).from_pretrained("tartuNLP/EstBERT")
model = BertModel(config)

words_tensor = torch.tensor([[(1,2),(3,2),(3,5),(4,5)]])
segments_tensors = torch.tensor([[1,1,1,1]])

print(model(words_tensor, segments_tensors))

#print(embedding.parameters)

tokenizer_output = tokenizer2(input_tekst)
words_tensor = torch.tensor([tokenizer_output["input_ids"]])
segments_tensor = torch.tensor([tokenizer_output["token_type_ids"]])

print(model(words_tensor, segments_tensor))
