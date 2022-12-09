from src.transformers.models.bert.tokenization_bert import BertTokenizer
from src.transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertForMaskedLM
import torch
from estnltk import Text
from estnltk.corpus_processing.parse_enc import parse_enc_file_iterator
from src.transformers.models.bert.configuration_bert import BertConfig
from transformers import AdamW
from tqdm.auto import tqdm
from transformers import pipeline

tokenizer = BertTokenizer(vocab_file = "vocab.txt", vocab_file_form = "vocab_form.txt")

# Korpus
input_file = "estonian_nc17.vert"
n = 3 # Mitu teksti korpusesse lugeda
korpus = []
l = 0
for text_obj in parse_enc_file_iterator(input_file):
    korpus.append(text_obj.text)
    if l > n:
        break
    l += 1

tekst = Text(" ".join(korpus)).tag_layer()
laused = []
for span in tekst.sentences:
    laused.append(tekst.text[span.start:span.end])
train =  laused[:int(0.8*len(laused))]
test = laused[int(0.8*len(laused)):]

# Maskimine
def mlm(tensor):
    rand = torch.rand(tensor[:, :, 0].shape)
    mask_arr = (rand < 0.15) * (tensor[:, :, 0] > 5)
    for i in range(tensor[:, :, 0].shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i, selection] = 4 # [MASK] tokeni ID mõlemas vocabis
    return tensor

input_ids = []
mask = []
labels = []

#print(tokenizer(train[0]))
#print(tokenizer.create_token_type_ids_from_sequences([(2,1),(5,3),(4,4)]))
#print(tokenizer._pad(tokenizer(train[0])))

sample = tokenizer(train, max_length = 64, padding = "max_length", truncation = True, return_tensors = "pt")
labels.append(sample.input_ids[:, :, 1]) # Labeliks võtame sõnavormi id
mask.append(sample.attention_mask)
input_ids.append(mlm(sample.input_ids.detach().clone()))

input_ids = torch.cat(input_ids)
mask = torch.cat(mask)
labels = torch.cat(labels)

encodings = {
    "input_ids" : input_ids,
    "attention_mask" : mask,
    "labels" : labels
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, i):
        return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)
dataloader_train = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)

config = BertConfig(
    vocab_size = tokenizer.vocab_size,
    vocab_size_form = tokenizer.vocab_size_form,
    tie_word_embeddings = False
)

model = BertForMaskedLM(config)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
optim = AdamW(model.parameters(), lr = 1e-4)
epochs = 2

for epoch in range(epochs):
    loop = tqdm(dataloader_train, leave = True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask = mask, labels = labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f"Epoch: {epoch}")
        loop.set_postfix(loss = loss.item())

#fill = pipeline("fill-mask", model = model, tokenizer = tokenizer)
#fill(f"Ma käisin Tallinna {fill.tokenizer.mask_token} platsil meelt avaldamas.")

input_ids2 = []
mask2 = []
labels2 = []

sample2 = tokenizer(test, max_length = 64, padding = "max_length", truncation = True, return_tensors = "pt")
labels2.append(sample2.input_ids[:, :, 1]) # Labeliks võtame sõnavormi id
mask2.append(sample2.attention_mask)
input_ids2.append(mlm(sample2.input_ids.detach().clone()))

input_ids2 = torch.cat(input_ids2)
mask2 = torch.cat(mask2)
labels2 = torch.cat(labels2)


encodings2 = {
    "input_ids" : input_ids2,
    "attention_mask" : mask2,
    "labels" : labels2
}
dataset2 = Dataset(encodings2)
dataloader_test = torch.utils.data.DataLoader(dataset2, batch_size = 16)

loop2 = tqdm(dataloader_test, leave = True)
for batch in loop2:
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids, attention_mask=mask, labels=labels)
    for i in outputs:
        print(outputs[i])


#model.eval()
#sample = tokenizer(test, max_length = 64, padding = "max_length", truncation = True, return_tensors = "pt")
#output = model(sample)