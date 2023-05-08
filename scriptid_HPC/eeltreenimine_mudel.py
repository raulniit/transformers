# Impordid
import os
from datasets import load_dataset
from src.transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datetime import datetime
import torch

print("1. Scripti käivitamine:")
print(datetime.now())
algus = datetime.now()

print("Cuda: ")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Sisendfailide nimed
input_folder = "korpus2/korpus"
input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename[-4:] == "json"]
print(input_files)

# Tokeniseerija
tokenizer = BertTokenizer(vocab_file = "vocab_final.txt", vocab_file_form = "vocab_form.txt", max_length = 128,
                          padding = "max_length", truncation = True, return_tensors = "pt", mask_token="ˇMASKˇ")

# Dataseti sisselugemine
treening_andmed = load_dataset("json", data_files={'train': input_files})["train"]
print("2. Andmestik sisse loetud")
print(datetime.now())

treening_andmed.set_format(type ='torch')

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
model = model.to(device)

training_args = TrainingArguments(
    output_dir='./train_results_mudel4',
    per_device_train_batch_size=32,
    max_steps=900000,
    learning_rate=1e-4,
    logging_dir='./train_logs_mudel4',
    logging_steps=50000,
    save_steps=50000,
    warmup_steps=9000
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator = data_collator,
    train_dataset=treening_andmed
)

# https://github.com/nlp-with-transformers/notebooks/issues/31
old_collator = trainer.data_collator
trainer.data_collator = lambda data: dict(old_collator(data))

trainer.train()

print("4. Treenimine lõpetatud")
print(datetime.now())
lopp = datetime.now()
print("Kestus")
print(lopp-algus)