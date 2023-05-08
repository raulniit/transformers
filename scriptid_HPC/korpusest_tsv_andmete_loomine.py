# Impordid
from estnltk import Text
from estnltk.corpus_processing.parse_enc import parse_enc_file_iterator
import os
import csv
from datetime import datetime


input_folder = "korpus"

# Käime läbi kõik korpuse failid ning kirjutame kõik dokumendid koos lausetega välja
# Dokumendi laused on eraldatud sümboliga "\n" ning dokumendid omavahel sümboliga "\t"
# Iga korpuse faili kohta tekib .tsv fail, mida on edaspidi lihtsam töödelda -> treeningandmete_loomine.py scriptis
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    read = []
    for text_obj in parse_enc_file_iterator(file_path):
        if "lang" in text_obj.meta:
            if text_obj.meta["lang"] != "Estonian":
                continue
        laused = []
        for span in text_obj.original_sentences:
            laused.append(text_obj.text[span.start:span.end])
        read.append("\n".join(laused))
    with open(file_path + ".tsv", 'w', newline='', encoding="utf-8") as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for record in read:
            writer.writerow([record])
    print(file_path)
    print(datetime.now())

