#!/bin/python

# Import Statements
import re
import sys
import string
from nltk.corpus import udhr

# A global value to be considered if the llanguage model
# fails to search for a charahter
smoothingValue = 0.000001


# Funtion to remove punctuation from training data
# Input    : data    - data to be cleaned
# Output   : Input data free of punctuation and new line
#            charachters in LowerCase

def cleanData(data):
    punct = string.punctuation
    data = re.sub('[\n]', '', data).lower()
    return "".join([x for x in data if x not in punct])


# Function that generates pairs of data
# Input    : data    - data for which the pairs are to be generated
#            length  - Number of charachters in the tuples generated
#                      Default value is 2 and can accept only 2 or 3.
# Output   : Returns a list of tuples of data generated from the input data
#            of length specified in the 2nd parameter, length

def __genPairs(data, length=2):
    tempList = []
    if length == 3:
        for i in range(len(data) - 2):
            tempList.append((data[i], data[i+1], data[i+2]))
    elif length == 2:
        for i in range(len(data) - 1):
            tempList.append((data[i], data[i+1]))
    else:
        print ("This function generates pairs of sizes 2 and 3 only!")
    return tempList


# Function to build a UniGram
# Input   : data   - Data on which the TriGram model is to be built
# Output  : Returns a dictionary along with the
#           probabilities for the corresponding charachters

def buildUniGram(data):
    uniGram = {}
    normalizer = len(data)
    for i in data:
        uniGram[i] = len(re.findall(i, data))/float(normalizer)
    return uniGram


# Function to generate BiGram from the given data
# Input   : data   - Data on which the TriGram model is to be built
# Output  : Returns a dictionary with 2 levels along with the
#           probabilities for the corresponding charachters

def buildBiGram(data):
    biGramPairs = __genPairs(data)
    biFreqDist = {}
    for i, j in biGramPairs:
        if biFreqDist.get(i) is not None:
            biFreqDist[i].append((j,
                                  len(re.findall(i+j, data))/float(
                                      len(re.findall(i, data)))))
        else:
            biFreqDist[i] = [(j,
                              len(re.findall(i+j, data))/float(
                                  len(re.findall(i, data))))]
    for x, y in biFreqDist.items():
        biFreqDist[x] = dict(y)
    return biFreqDist


# Function to generate TriGram from the given data
# Input   : data   - Data on which the TriGram model is to be built
# Output  : Returns a dictionary with 3 levels along with the
#           probabilities for the corresponding charachters

def buildTriGram(data):
    triGramPairs = __genPairs(data, 3)
    triFreqDist = {}
    for i, j, k in triGramPairs:
        res1 = float(len(re.findall(i+j+k, data)))
        res2 = float(len(re.findall(i+j, data)))
        if i in triFreqDist.keys() and j in triFreqDist[i].keys():
            triFreqDist[i][j].append((k, res1/res2))
        elif i in triFreqDist.keys():
            triFreqDist[i][j] = [(k, res1/res2)]
        else:
            triFreqDist[i] = {j: [(k, res1/res2)]}

    for x in triFreqDist.items():
        for y in x[1:]:
            for k, v in y.items():
                y[k] = dict(v)
    return triFreqDist


# Function to predict the language of test data using UniGrams
# Input   :  uniGram1   -  UniGram model 1
#            uniGram2   -  UniGram model 2
#            testData   -  Data on which the models are to be tested
# Output  :  Returns accuracy of uniGram1 for the test data supplied

def predictUnigram(uniGram1, uniGram2, testData):
    positiveCount = 0
    for word in testData:
        uni1Prob = uni2Prob = 1
        for char in word:
            if uniGram1.get(char) is not None:
                uni1Prob *= uniGram1[char]
            else:
                uni1Prob *= smoothingValue
            if uniGram2.get(char) is not None:
                uni2Prob *= uniGram2[char]
            else:
                uni2Prob *= smoothingValue
        if uni1Prob >= uni2Prob:
            positiveCount += 1
    return float(positiveCount*100)/len(testData)


# Function to calculate the initial probabilities for BiGram prediction
# Input   : uniGram1  - Unigram model
#           word      - A tupel of charachters
# Output  : Returns probability based on the Fall Back Strategy.
#           If one of the values of the tupel is not in unigram model passed,
#           it returns smoothingvalue

def __calcBiInitProb(uniGram1, word):
    if uniGram1.get(word[0]) is not None and uniGram1.get(word[1]) is not None:
        return (uniGram1.get(word[0])) * (uniGram1.get(word[1]))
    elif uniGram1.get(word[0]) is not None:
        return smoothingValue * uniGram1.get(word[0])
    elif uniGram1.get(word[1]) is not None:
        return smoothingValue * uniGram1.get(word[1])
    else:
        return smoothingValue * smoothingValue


# Function to predict the language of test data using BiGrams
# Input  : biGram1   - biGram model 1
#          biGram2   - biGram model 2
#          UniGram1  - UniGram model 1
#          UniGram2  - UniGram model 2
#          testData  - Testing Data
#
# Output : Returns accuracy of language predicted by biGram1 and biGram2

def predictBigram(biGram1, biGram2, uniGram1, uniGram2, testData):
    positiveCount = 0
    for word in testData:
        bi1Prob = bi2Prob = 1
        # Words of length 1 are assigned UniGram probabilities
        if (len(word) == 1):
            bi1Prob = uniGram1.get(word[0]) or smoothingValue
            bi2Prob = uniGram2.get(word[0]) or smoothingValue
            if bi1Prob >= bi2Prob:
                positiveCount += 1
            continue
        # Initial probability is the product of first and last letters
        # to support P(x/<s>) and P(</s>/x) respectively
        bi1Prob = __calcBiInitProb(uniGram1, (word[0], word[-1]))
        bi2Prob = __calcBiInitProb(uniGram2, (word[0], word[-1]))

        for i in range(len(word)-1):
            if biGram1.get(word[i]) and biGram1[word[i]].get(word[i+1]):
                bi1Prob *= biGram1[word[i]][word[i+1]]
            else:
                bi1Prob *= __calcBiInitProb(uniGram1, (word[i], word[i+1]))
            if biGram2.get(word[i]) and biGram2[word[i]].get(word[i+1]):
                bi2Prob *= biGram2[word[i]][word[i+1]]
            else:
                bi2Prob *= __calcBiInitProb(uniGram2, (word[i], word[i+1]))
        # The word is inclined to language of the 1st model passed.
        if bi1Prob >= bi2Prob:
            positiveCount += 1
    return float(positiveCount*100)/len(testData)


# Function to calculate the probabilities for TriGram when the sequence is
# not present in TriGram
# Input  : triGram   - triGram model
#          biGram    - biGram model
#          UniGram   - UniGram model
#          testData  - Testing Data
#
# Output : Returns probability based on the Fall Back Strategy.
#          ie., returns bigram probabilities if data is not found in trigram
#          or returns unigram probabilities when not in bigram
#          or returns smoothingvalue when data is not found in any of the
#        language models

def __getTriProb(triGram, biGram, uniGram, data):
    if triGram.get(data[0]) and triGram[data[0]].get(data[1])  \
            and triGram[data[0]][data[1]].get(data[2]):
        return triGram[data[0]][data[1]][data[2]]
    elif biGram.get(data[1]) and biGram.get(data[0]):
        return (biGram.get(data[1]).get(data[2]) or smoothingValue)  \
                * (biGram.get(data[0]).get(data[1]) or smoothingValue)
    elif biGram.get(data[1]) and biGram.get(data[0]) is None:
        return ((biGram.get(data[1]).get(data[2]) or smoothingValue)
                * smoothingValue)
    elif biGram.get(data[0]) and biGram.get(data[1]) is None:
        return ((biGram.get(data[0]).get(data[1]) or smoothingValue)
                * smoothingValue)
    else:
        return (uniGram.get(data[0]) or smoothingValue)*(uniGram.get(data[1])
                  or smoothingValue) * (uniGram.get(data[2]) or smoothingValue)


# Function to calculate the initial probabilities for TriGram prediction
# Input  : biGram1    - BiGram model
#          UniGram1   - UniGram model
#          word       - A tupel whose initial probabilities are to be calculated
# Output : Returns probability based on Fall Back strategy
#          ie., returns bigram probabilities when found in bigram language model
#          or returns unigram probabilities when not in bigram
#          or returns smoothingvalue when data is not found in any of the
#          language models

def __calcInitProb(biGram1, uniGram1, word):
    if biGram1.get(word[0]) and biGram1[word[0]].get(word[1]):
        return biGram1[word[0]][word[1]]
    else:
        return __calcBiInitProb(uniGram1, word)


# Function to predict the language of test data using TriGrams
# Input  : triGram1   - triGram model 1
#          triGram2   - triGram model 2
#          biGram1    - biGram model 1
#          biGram2    - biGram model 2
#          UniGram1   - UniGram model 1
#          UniGram2   - UniGram model 2
#          testData   - Testing Data
#
# Output : Returns accuracy of language predicted by triGram1 and triGram2

def predictTrigram(triGram1, triGram2, biGram1, biGram2, uniGram1, uniGram2,
                   testData):
    positiveCount = 0
    for word in testData:
        tri1Prob = tri2Prob = 1
        if (len(word) == 2):
            tri1Prob = __calcInitProb(biGram1, uniGram1, (word[0], word[1]))
            tri2Prob = __calcInitProb(biGram2, uniGram2, (word[0], word[1]))
            if tri1Prob >= tri2Prob:
                positiveCount += 1
            continue
        elif (len(word) == 1):
            tri1Prob = uniGram1.get(word) or smoothingValue
            tri2Prob = uniGram2.get(word) or smoothingValue
            if tri1Prob >= tri2Prob:
                positiveCount += 1
            continue

        tri1Prob = __calcInitProb(biGram1, uniGram1, (word[0], word[1]))
        tri2Prob = __calcInitProb(biGram2, uniGram2, (word[0], word[1]))

        for i in range(len(word)-2):
            tri1Prob *= __getTriProb(triGram1, biGram1, uniGram1, word[i:i+3])
            tri2Prob *= __getTriProb(triGram2, biGram2, uniGram2, word[i:i+3])
        if tri1Prob >= tri2Prob:
            positiveCount += 1
    return float(positiveCount*100)/len(testData)


# Function to build UniGrams, BiGrams and TriGrams and comparision of the built
# models for various languages
# Input   : None
# Output  : None

def main():
    # Importing Corpus from NLTK
    english = udhr.raw('English-Latin1')
    french = udhr.raw('French_Francais-Latin1')
    italian = udhr.raw('Italian_Italiano-Latin1')
    spanish = udhr.raw('Spanish_Espanol-Latin1')

    # Training and Development Data
    english_train = cleanData(english[0:1000])
    french_train  = cleanData(french[0:1000])
    italian_train = cleanData(italian[0:1000])
    spanish_train = cleanData(spanish[0:1000])

    # Preparing Test Data
    english_test = udhr.words('English-Latin1')[0:1000]
    french_test = udhr.words('French_Francais-Latin1')[0:1000]
    italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000]
    spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

    eUniGram = buildUniGram(english_train)
    fUniGram = buildUniGram(french_train)
    iUniGram = buildUniGram(italian_train)
    sUniGram = buildUniGram(spanish_train)

    eBiGram = buildBiGram(english_train)
    fBiGram = buildBiGram(french_train)
    iBiGram = buildBiGram(italian_train)
    sBiGram = buildBiGram(spanish_train)

    eTriGram = buildTriGram(english_train)
    fTriGram = buildTriGram(french_train)
    iTriGram = buildTriGram(italian_train)
    sTriGram = buildTriGram(spanish_train)

    print ("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print ("$                    English  Vs French                    $")
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    print ("--------------------------------------------------")
    print ("|                   UniGrams                     |")
    print ("--------------------------------------------------")
    print ("TestData\Model\t\tEnglish\t\tFrench")
    print ("English Test   \t\t" + str(predictUnigram(eUniGram, fUniGram,
            english_test)) + "%  \t\t" + str(predictUnigram(fUniGram,
            eUniGram, english_test)) + "%")
    print ("French Test    \t\t"+str(predictUnigram(eUniGram, fUniGram \
            ,french_test))+"%  \t\t" +str(predictUnigram(fUniGram, \
            eUniGram ,french_test))+"%")
    print ("\n\n--------------------------------------------------")
    print ("|                    BiGrams                     |")
    print ("--------------------------------------------------")
    print ("TestData\Model\t\tEnglish\t\tFrench")
    print ("English Test   \t\t" +
            str(predictBigram(eBiGram, fBiGram ,eUniGram, fUniGram,
            english_test)) + "%  \t\t"
            + str(predictBigram(fBiGram, eBiGram ,fUniGram, eUniGram\
             ,english_test)) + "%")
    print ("French Test    \t\t"+\
            str(predictBigram(eBiGram, fBiGram ,eUniGram, fUniGram\
             ,french_test)) + "%  \t\t"\
            +str(predictBigram(fBiGram, eBiGram ,fUniGram, eUniGram \
            ,french_test)) + "%")
    print ("\n\n--------------------------------------------------")
    print ("|                   TriGrams                     |")
    print ("--------------------------------------------------")
    print ("TestData\Model\t\tEnglish\t\tFrench")
    print ("English Test   \t\t"+
            str(predictTrigram(eTriGram, fTriGram, eBiGram, fBiGram,
            eUniGram, fUniGram ,english_test)) + "%  \t\t" +
            str(predictTrigram(fTriGram, eTriGram, fBiGram, eBiGram \
            ,fUniGram, eUniGram ,english_test)) + "%")
    print ("French Test    \t\t" +
            str(predictTrigram(eTriGram, fTriGram, eBiGram, fBiGram
            ,eUniGram, fUniGram ,french_test)) +
            "%  \t\t" + str(predictTrigram(fTriGram, eTriGram, fBiGram
            ,eBiGram ,fUniGram, eUniGram ,french_test)) + "%")
    print ("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print ("$                    Spanish Vs Italian                    $")
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    print ("--------------------------------------------------")
    print ("|                   UniGrams                     |")
    print ("--------------------------------------------------")
    print ("TestData\Model\t\tSpanish\t\tIatlian")
    print ("Spanish Test   \t\t"+str(predictUnigram(sUniGram, iUniGram
            ,spanish_test))+"%  \t\t" + str(predictUnigram(iUniGram,
            sUniGram ,spanish_test)) + "%")
    print ("Italian Test   \t\t"+str(predictUnigram(sUniGram, iUniGram
            ,italian_test))+"%  \t\t" +str(predictUnigram(iUniGram,
            sUniGram ,italian_test)) + "%")
    print ("\n\n--------------------------------------------------")
    print ("|                    BiGrams                     |")
    print ("--------------------------------------------------")
    print ("TestData\Model\t\tSpanish\t\tIatlian")
    print ("Spanish Test   \t\t" +
            str(predictBigram(sBiGram, iBiGram ,sUniGram, iUniGram,
            spanish_test))+"%  \t\t" + str(predictBigram(iBiGram, sBiGram,
                                                         iUniGram, sUniGram,
                                                         spanish_test))+"%")
    print ("Italian Test   \t\t"+\
            str(predictBigram(iBiGram, sBiGram, iUniGram, sUniGram\
             ,italian_test))+"%  \t\t"\
            +str(predictBigram(iBiGram, sBiGram, iUniGram, sUniGram \
            ,italian_test))+"%")
    print ("\n\n--------------------------------------------------")
    print ("|                   TriGrams                     |")
    print ("--------------------------------------------------")
    print ("TestData\Model\t\tSpanish\t\tItalian")
    print ("Spanish Test   \t\t"+\
            str(predictTrigram(sTriGram, iTriGram, sBiGram, iBiGram \
            ,sUniGram, iUniGram ,spanish_test))+\
            "%  \t\t"+\
            str(predictTrigram(iTriGram, sTriGram, iBiGram, sBiGram \
            ,iUniGram, sUniGram ,spanish_test)) +"%")
    print ("Italian Test   \t\t"+\
            str(predictTrigram(sTriGram, iTriGram, sBiGram, iBiGram \
            ,sUniGram, iUniGram ,italian_test))+\
            "%  \t\t"+str(predictTrigram(iTriGram, sTriGram, iBiGram\
            ,sBiGram ,iUniGram, sUniGram ,italian_test))\
            +"%")


# Main starts  here

if __name__ == "__main__":
    if len(sys.argv) > 1 and float(sys.argv[1]) == 0:
        smoothingValue = 0
    print ("Smoothing Vlaue : " + str(smoothingValue))
    main()
