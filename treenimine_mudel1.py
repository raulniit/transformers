# Impordid
import os
from datasets import load_dataset
from src.transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datetime import datetime

print("1. Scripti käivitamine:")
print(datetime.now())
algus = datetime.now()

# Sisendfailide nimed
input_folder = "korpus"
input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename[-3:] == "tsv"]
print(input_files)
# Tokeniseerija
tokenizer = BertTokenizer(vocab_file = "vocab_final.txt", vocab_file_form = "vocab_form.txt", max_length = 128, padding = "max_length", truncation = True, return_tensors = "pt")

# Dataseti sisselugemine
train_dataset = load_dataset("csv", data_files={'train': input_files[0]}, names = ["text"], delimiter = "\t")["train"]

print("2. Andmestik loetud")
print(datetime.now())

# Võtab iga batchi (tekstid korpusest, kus laused on eraldatud \n sümboliga, 'text' : lause1 \n lause2 \n lause3)
# Tokenieerib laused ja lisab batchile labelid
# Väjund (x tähistab kõikide lausete arvu loetud batchi tekstides)
# {'input_ids': x*128*2 tensor, 'attention_mask': x*128 tensor, 'token_type_ids': x*128 tensor, 'labels': x*128*2 tensor}
def tokeniseeri_batch(batch):
    batch_laused = [lause for para in batch["text"] for lause in para.split("\n")]
    tokeniseeritud = tokenizer(batch_laused, max_length = 128, padding = "max_length", truncation = True, return_tensors = "pt")
    tokeniseeritud["labels"] = tokeniseeritud["input_ids"]
    return tokeniseeritud

train_dataset_sisend = train_dataset.map(tokeniseeri_batch, batched=True, remove_columns=["text"])
train_dataset_sisend.set_format(type ='torch')

print("3. Andmestik treenimiseks ette valmistatud")
print(datetime.now())

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
)

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
    learning_rate=1e-4,
    logging_dir='./train_logs',
    logging_steps=100,
    save_steps=100,
    warmup_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator = data_collator,
    train_dataset=train_dataset_sisend
)

trainer.train()

print("4. Treenimine lõpetatud")
print(datetime.now())
lopp = datetime.now()
print("Kestus")
print(lopp-algus)