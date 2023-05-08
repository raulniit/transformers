# Impordid
import torch
import pandas as pd
import numpy as np
from src.transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.metrics import classification_report
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime

print("1. Scripti käivitamine:")
print(datetime.now())
algus = datetime.now()

andmed = pd.read_csv("Rubric_data/estonianvalence.csv", encoding = "utf8", on_bad_lines='skip', header = None,
                     names = ["rubric","url", "order", "sentiment", "text"])

print(andmed.rubric.value_counts())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train, test = train_test_split(andmed, test_size=0.2)
train, val = train_test_split(train, test_size=0.125)

tokenizer = BertTokenizer(vocab_file = "vocab_final.txt", vocab_file_form = "vocab_form.txt", max_length = 128,
                         padding = "max_length", truncation = True, return_tensors = "pt", mask_token="ˇMASKˇ")

print(train.rubric.value_counts())
print(val.rubric.value_counts())
print(test.rubric.value_counts())

train_encodings = tokenizer(list(train.text), max_length = 128, padding = "max_length",
                            truncation = True, return_tensors = "pt")
train_dataset = Dataset.from_dict(train_encodings)

val_encodings = tokenizer(list(val.text), max_length = 128, padding = "max_length",
                            truncation = True, return_tensors = "pt")
val_dataset = Dataset.from_dict(val_encodings)

test_encodings = tokenizer(list(test.text), max_length = 128, padding = "max_length",
                            truncation = True, return_tensors = "pt")
test_dataset = Dataset.from_dict(test_encodings)

print("2. Andmestik treenimiseks ette valmistatud")
print(datetime.now())

kaustad = ["train_results_mudel4/checkpoint-100000", "train_results_mudel4/checkpoint-200000", "train_results_mudel4_round2/checkpoint-50000", "train_results_mudel4_round2/checkpoint-150000", "train_results_mudel4_round2/checkpoint-250000"]

for i in range(len(kaustad)):

    model = BertForMaskedLM.from_pretrained(kaustad[i])
    model.to(device)

    batch_size = 16

    def compute_metrics(p):
        predictions, labels = p
        predictions_lemma = np.argmax(predictions[0], axis=2).ravel()
        labels_lemma = [y[0] for x in labels for y in x]
        predictions_vorm = np.argmax(predictions[1], axis=2).ravel()
        labels_vorm = [y[1] for x in labels for y in x]

        final_pred_lemma = [(p, l) for p, l in zip(predictions_lemma, labels_lemma) if l != -100]
        acc_lemma = sum(np.array([x[0] for x in final_pred_lemma]) == np.array([x[1] for x in final_pred_lemma])) / len(
            final_pred_lemma)

        final_pred_vorm = [(p, l) for p, l in zip(predictions_vorm, labels_vorm) if l != -100]
        acc_vorm = sum(np.array([x[0] for x in final_pred_vorm]) == np.array([x[1] for x in final_pred_vorm])) / len(
            final_pred_vorm)

        return ({'Accuracy_lemma': acc_lemma, 'Accuracy_vorm': acc_vorm, 'n_val': len(final_pred_lemma)})


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    args = TrainingArguments(
        "MLM_results_" + str((i+1)*100000),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3
    )

    trainer = Trainer(
        model,
        args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.evaluate()
    trainer.train()

    predictions, labels, _ = trainer.predict(test_dataset)

    predictions_lemma = np.argmax(predictions[0], axis=2).ravel()
    labels_lemma = [y[0] for x in labels for y in x]
    predictions_vorm = np.argmax(predictions[1], axis=2).ravel()
    labels_vorm = [y[1] for x in labels for y in x]

    final_pred_lemma = [(p, l) for p, l in zip(predictions_lemma, labels_lemma) if l != -100]
    acc_lemma = sum(np.array([x[0] for x in final_pred_lemma]) == np.array([x[1] for x in final_pred_lemma])) / len(final_pred_lemma)

    final_pred_vorm = [(p, l) for p, l in zip(predictions_vorm, labels_vorm) if l != -100]
    acc_vorm = sum(np.array([x[0] for x in final_pred_vorm]) == np.array([x[1] for x in final_pred_vorm])) / len(final_pred_vorm)

    print(f"Accuracy lemma : {acc_lemma}")
    print(f"Accuracy vorm : {acc_vorm}")
    print(f"n test : {len(final_pred_lemma)}")
    pd.DataFrame(final_pred_lemma).to_csv(str((i+1)*100000) + "test_lemma.csv")
    pd.DataFrame(final_pred_vorm).to_csv(str((i+1)*100000) + "test_vorm.csv")

print("4. Script lõpetatud")
print(datetime.now())
lopp = datetime.now()
print("Kestus")
print(lopp-algus)