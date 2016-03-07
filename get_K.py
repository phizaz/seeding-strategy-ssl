from dataset import *

# last run likelihood cv -- 1e-4 -> 1.0 (1000) cv = 10 folds rtol 1e-6

print('working on iris')
iris = get_iris()
print('iris K:', iris.K_for_KNN)

print('working on yeast')
yeast = get_yeast()
print('yeast K:', yeast.K_for_KNN)

print('working on pendigits')
pendigits = get_pendigits()
print('pendigits K:', pendigits.K_for_KNN)

print('working on satimage')
satimage = get_satimage()
print('satimage K:', satimage.K_for_KNN)

print('working on banknote')
banknote = get_banknote()
print('banknote K:', banknote.K_for_KNN)

print('working on eeg')
eeg = get_eeg()
print('eeg K:', eeg.K_for_KNN)

print('working on magic')
magic = get_magic()
print('magic K:', magic.K_for_KNN)

print('working on spam')
spam = get_spam()
print('spam K:', spam.K_for_KNN)

print('working on letter')
letter = get_letter()
print('letter K:', letter.K_for_KNN)

print('all K generated')