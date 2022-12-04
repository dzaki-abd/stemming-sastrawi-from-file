# Library
import fitz
import nltk
import re  # Menghapus karakter angka.
import string  # Menghapus karakter tanda baca.
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Input Dari PDF ke String
with fitz.open("example.pdf") as doc:
    kalimat = ""
    for page in doc:
        kalimat += page.get_text()


# kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."

# Merubah format teks menjadi format huruf kecil semua (lowercase).
lower_case = kalimat.lower()

# Menghapus karakter angka.
remove_number = re.sub(r"\d+", "", lower_case)

# Menghapus karakter tanda baca.
remove_punctuation = remove_number.translate(str.maketrans("","",string.punctuation))

# Menghapus karakter kosong.
removing_whitespace = remove_punctuation.strip()

# Menggunakan library NLTK untuk memisahkan kata dalam sebuah kalimat.
tokens = nltk.tokenize.word_tokenize(removing_whitespace)

hasil_tokens = ''
for x in tokens:
    hasil_tokens += ' ' + x

# Menghapus kata Stopword - Sastrawi
stop_factory = StopWordRemoverFactory().get_stop_words()  # load default stopword
more_stopword = ['vol', 'volume', 'issn', 'php', 'mysql', 'gunadarma']  # menambahkan stopword

data = stop_factory + more_stopword  # menggabungkan stopword

dictionary = ArrayDictionary(data)
str_stopword = StopWordRemover(dictionary)
hasil_stopword = nltk.tokenize.word_tokenize(str_stopword.remove(hasil_tokens))

hasil_stopword_fix = ''
for x in hasil_stopword:
    hasil_stopword_fix += ' ' + x

# Stemming Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

hasil = stemmer.stem(hasil_stopword_fix)

def Convert(string):
    li = list(string.split(" "))
    return li

# Menghitung frekuensi kemunculan setiap tokens(kata) dalam teks.
kemunculan = nltk.FreqDist(Convert(hasil))

# Menggambarkan frekuensi kemunculan setiap tokens
kemunculan.plot(100, cumulative=False)
plt.show()