from src.transformers.tokenization_utils import PreTrainedTokenizer
from src.transformers.models.bert.tokenization_bert import BertTokenizer
from src.transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
from src.transformers.models.bert.configuration_bert import BertConfig
import torch

tokenizer = PreTrainedTokenizer()
tokenizer2 = BertTokenizer(vocab_file = "vocab.txt", vocab_file_form = "vocab_form.txt", never_split = ["[MASK]"])

input_tekst = "Maril on paha [MASK], aga see on [MASK] ja ta elab Tallinnas."
#input_tekst ='Tallinna [MASK] algatab Paldiski maantee ääres Hotell Tallinna kõrval asuva vundamendiaugu.'

#print(tokenizer.tokenize(input_tekst))
#print(tokenizer(input_tekst))
# print(tokenizer.added_tokens_encoder)
#print()
print(tokenizer2.tokenize(input_tekst))
print(tokenizer2(input_tekst))
print(tokenizer2.decode(25, return_form = True))
config = BertConfig()
embedding = BertEmbeddings(config)
#model = BertModel(config).from_pretrained("tartuNLP/EstBERT")
model = BertModel(config)

#words_tensor = torch.tensor([[(1,2),(3,2),(3,5),(4,5)]])
#segments_tensors = torch.tensor([[1,1,1,1]])
#print(model(words_tensor, segments_tensors))
#print(embedding.parameters)

tokenizer_output = tokenizer2(input_tekst)
words_tensor = torch.tensor([tokenizer_output["input_ids"]])
segments_tensor = torch.tensor([tokenizer_output["token_type_ids"]])

#print(model(words_tensor, segments_tensor))
