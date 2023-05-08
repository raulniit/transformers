from estnltk.corpus_processing.parse_enc import parse_enc_file_iterator
import os
from operator import itemgetter
from datetime import datetime

input_folder = "korpus"

lemmad = dict()


# Käime läbi kõik korpuse failid ning loendame kokku lemmad dict abil
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    print(filename)
    l = 0
    for text_obj in parse_enc_file_iterator(file_path, restore_morph_analysis = True):
        if "lang" in text_obj.meta:
            if text_obj.meta["lang"] != "Estonian":
                continue
        for lemma in [lem[0] for lem in text_obj["original_morph_analysis"]["lemma"]]:
            if lemma in lemmad.keys():
                lemmad[lemma] = lemmad[lemma] + 1 # Kui lemma on sõnastikus suurendame loendust ühe võrra
            else:
                lemmad[lemma] = 1 # Kui lemmat veel pole sõnastikus, lisatakse see juurde
        if l % 10000 == 0:
            print(l)
            t = datetime.now()
            print(t)
        l += 1

N = 50000 # Mitu kõige sagedasemat sõna leida
res = dict(sorted(lemmad.items(), key = itemgetter(1), reverse = True)[:N])
with open("dict.txt", "w", encoding = "utf8") as dict_file:
    dict_file.write(str(res))

with open("vocab_final.txt", "w", encoding = "utf8") as txt_file:
    for lemma in res.keys():
        txt_file.write(lemma + "\n")
