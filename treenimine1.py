# Impordid
import torch
from estnltk.corpus_processing.parse_enc import parse_enc_file_iterator
from src.transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer, TrainingArguments
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score

print("1. Scripti käivitamine:")
print(datetime.now())
algus = datetime.now()

# Tokeniseerija
tokenizer = BertTokenizer(vocab_file = "vocab_final.txt", vocab_file_form = "vocab_form.txt")

# Korpus
# Loeme tekstid sisse, laseme estnltk-l laused leida ning moodustame lausetest treening- ja testhulg
# https://github.com/estnltk/estnltk/blob/main/tutorials/corpus_processing/importing_text_objects_from_corpora.ipynb

input_folder = "korpus"
korpus = []

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    n = 1000 # Mitu teksti korpusesse lugeda
    l = 0
    for text_obj in parse_enc_file_iterator(file_path):
        korpus.append(text_obj)
        if l > n:
            break
        l += 1

train =  korpus[:int(0.6*len(korpus))]
val =  korpus[int(0.6*len(korpus)):int(0.8*len(korpus))]
test = korpus[int(0.8*len(korpus)):]

train_laused = []
for tekst in train:
    for span in tekst.original_sentences:
        train_laused.append(tekst.text[span.start:span.end])

val_laused = []
for tekst in val:
    for span in tekst.original_sentences:
        val_laused.append(tekst.text[span.start:span.end])

test_laused = []
for tekst in test:
    for span in tekst.original_sentences:
        test_laused.append(tekst.text[span.start:span.end])

print("2. Korpus loodud")
print(datetime.now())

# Dataseti loomine dataloaderi jaoks
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, i):
        return {key: tensor[i] for key, tensor in self.encodings.items()}


# Maskimine ja labelid
def mlm(tensor):
    labels = tensor.detach().clone()
    rand = torch.rand(tensor[:, :, 0].shape)
    mask_arr = (rand < 0.15) * (tensor[:, :, 0] > 5)
    for i in range(tensor[:, :, 0].shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        selection_masked = torch.where(mask_arr[i] == 0)[0].tolist()
        tensor[i, selection] = 4  # [MASK] tokeni ID mõlemas vocabis
        labels[i, selection_masked] = -100  # Et mudel arvutaks lossi ainult masked tokenite pealt

    return tensor, labels


# Andmestiku ettevalmistamine treenimiseks või mudelis kasutamiseks
def prepare_data(data):
    input_ids = []
    mask = []
    labels = []

    sample = tokenizer(data, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    new_ids, new_labels = mlm(sample.input_ids.detach().clone())
    input_ids.append(new_ids)
    mask.append(sample.attention_mask)
    labels.append(new_labels)

    input_ids = torch.cat(input_ids)
    mask = torch.cat(mask)
    labels = torch.cat(labels)

    encodings = {
        "input_ids": input_ids,
        "attention_mask": mask,
        "label_ids": labels
    }

    dataset = Dataset(encodings)
    return dataset


# Dataloaderi loomine mudeli treenimiseks
train_dataset = prepare_data(train_laused)
val_dataset = prepare_data(val_laused)
test_dataset = prepare_data(test_laused)

print("3. Andmestikud loodud")
print(datetime.now())

def compute_metrics(p):
    pred, labels = p

    indeksid = np.where(labels[:, :, 0].flatten() != -100)[0]

    labels_lemma = labels[:, :, 0].flatten()[indeksid]
    labels_vorm = labels[:, :, 1].flatten()[indeksid]

    pred_lemma = np.take(np.argmax(pred[0], axis=2).flatten(), indeksid)
    pred_vorm = np.take(np.argmax(pred[1], axis=2).flatten(), indeksid)

    accuracy_lemma = accuracy_score(y_true=labels_lemma, y_pred=pred_lemma)
    accuracy_vorm = accuracy_score(y_true=labels_vorm, y_pred=pred_vorm)

    return {"accuracy_lemma": accuracy_lemma, "accuracy_vorm": accuracy_vorm}

config = BertConfig(
    vocab_size = tokenizer.vocab_size,
    vocab_size_form = tokenizer.vocab_size_form,
    tie_word_embeddings = False
)

model = BertForMaskedLM(config)

training_args = TrainingArguments(
    output_dir='./train_results',
    per_device_train_batch_size=32,
    max_steps=1000,
    eval_steps=1000,
    learning_rate=1e-4,
    logging_steps=500,
    warmup_steps=100,
    save_steps=500,
    logging_dir='./train_logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

print("4. Treenimine lõpetatud")
print(datetime.now())
lopp = datetime.now()
print("Kestus")
print(lopp-algus)