Käesolev repositoorium on loodud Tartu Ülikooli andmeteaduse õppekava magistritöö "BERT mudeli kohandamine eesti keelele" raames. <br>
Töö autor on Raul Niit ning juhendajad Sven Laur ning Hendrik Šuvalov. <br>

Töö eesmärk oli uurida võimalusi BERT mudelile morfoloogilise info lisamiseks. <br>
See tähendab, et BERT mudeli sõnede asemel kasutatakse mudelis iga sõne lemmat ehk algvormi ning vormi ehk sõne käänet või pööret. <br>
Muudatuste implementeerimiseks tehti "fork" HuggingFace Transformers Pythoni paketi githubist.

Olulisemad osad repositooriumis:

* Kaust "Tulemused" - Töö käigus treenitud mudelite tulemused kohandamisülesannetel.
* Kaust "scriptid_HPC" - Sõnastiku loomiseks, andmete töötlemiseks ning mudeli eeltreenimiseks ja kohandamiseks kasutatud scriptid, mida jooksutati Tartu Ülikooli HPC keskuses (https://hpc.ut.ee/).
* Notebookid "test_" algusega - failid mudeli eri osade töötamise testimiseks.

Kaustast "korpus" on andmemahulise piirangu tõttu puudu Eesti keele 2017 ühendkorpuse failid, mis on kättesaadavad siit: https://doi.org/10.15155/3-00-0000-0000-0000-071E7L

Kui korpuse failid on olemas on töö scriptide loogiline järjekord 

1. vocabi_loomine.py - Mudeli sõnastiku loomine.
2. korpusest_tsv_andmete_loomine.py - Korpuse .vert failidest .tsv failide tekitamine.
3. treeningandmete_loomine.py - Töödeldud .tsv failide tokeniseerimine ning json kujule kirjutamine.
4. eeltreenimine_mudel.py - Mudeli eeltreenimine json failide põhjal.
5. MLM/NER/POS/Rubric_finetune.py - Mudeli kohandamine ülesande andmestikule (kaustad "NER_data", "conllu" ja "Rubric_data".
