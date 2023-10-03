import spacy
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

def createInput():
    '''
    Dataset are sample from sklearn which contain 18846 email sample
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
    For this project, only 20 samples are used 
    '''
    #Load the filenames and data from the 20 newsgroups dataset, filter only the body
    newsgroups = fetch_20newsgroups(subset = 'test', remove = ('footers','quotes'))
    with open('newsgroups.txt', 'w') as f:
        for mail in newsgroups.data[:100]:
            f.write(mail+'\n\n\n\n\n')

    
def tokenizeInput(textFile):
    '''
    Lemmatized input textfile and removed stop word
    Collect all lemmatized alphabetic legit word for words vector
    
    Input: text file instring format
    Output:
    -list of all lemmatized word
    -lemmatized text
    '''
    #load package "en_core_web_sm"
    nlp = spacy.load('en_core_web_sm')
    inputText = open(textFile, 'r', encoding='utf-8')
    textString = inputText.read()
    doc = nlp(textString)     #create token object with Doc
    #lemmatized text and remove stop word and number
    lemmatizedText=" ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.like_num])
    # filter only unique alphabetic word with vector represented and remove stop word then lemmatized it
    filterTokenizedWord = set(token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha and token.has_vector and not token.like_num)
    return filterTokenizedWord,lemmatizedText


#count frequent of unique lemmatized words for each lemmatized paragraph using Counter
def frequencyTokenVector(filterTokenizedWord,lemmatizedText):
    '''
    Create word vector for for each paragraph where the element represent how many time 
    the the word appear in paragraph
    
    Input: list of all lemmatized word, lemmatized text
    Output: list of dictionary where each dictionary represent a paragraph and its items represent all the words and their frequencies
    '''
    
    frequency_dict=[] #create frequency dict for each paragraph
    
    paragraphs=lemmatizedText.split('\n\n\n\n\n') #devide text into paragraph
    for paragraph in paragraphs[:-1]: #excluded the last empty paragraph
        #tokenize paragraph into word token
        wordTokens = paragraph.split()
        #calcualte word frequency
        wordTokenFrequency = Counter(wordTokens)
        #look up for frequency of all filterTokenizedWord in paragraph, for absent word, the frequency = 0
        filterTokenizedWordFrequency={word:wordTokenFrequency[word] for word in filterTokenizedWord}
        
        frequency_dict.append(filterTokenizedWordFrequency)
        
    return frequency_dict
    
    

def extractTopic(wordFrequencyParas, topCommonWords):
    '''
    Extract topic of each paragraph in text
    Topic are list of most common word in paragraph
    
    Input: dict contains word frequencies in paragrap, number of top common words
    Output: dictionary where key = paragraph index and values = dict(word,frequency)
    '''
    
    numParagraph=0 
    topicDict={}
    for wordFrequencyPara in wordFrequencyParas:
        topicDict[numParagraph]=dict(Counter(wordFrequencyPara).most_common(topCommonWords))
        numParagraph+=1
    return topicDict



def cosSim(a,b):
    """cal culate cosim similarity based on formular (A.B) / (||A||.||B||) where
    A.B the dot product and ||A|| is L2 norm of A (computed as square root of the sum of squares of elements of A)
    
    Input: vector a, vector b
    Output: cosSim similaritiy score
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)




def groupedTopic(wordFrequencyParas,scoreThreshold):
    
    '''Compare similarity of each paragraph using cosin similarity on word's vectors frequency 
    and put them in the same group when they have given cosSim score
    
    Input: dict contains word frequencies in paragrap, cosSim score 
    Output: dict contains paragraph index as key and values contain dict of other paragraph which 
    satifies the cosSim score and it score in format (paragraph_index, cosSin score)
    '''
    
    #create list of dictionaries where key = paragraph number and values = its frequency vector 
    paragrapFreqDict={}
    paragraphNum=0
    for wordFrequencyPara in wordFrequencyParas:
        paragrapFreqDict[paragraphNum]=list(wordFrequencyPara.values())
        paragraphNum+=1
    
    # create cossin dictionary with keys = paragraph numbers and values contain
    # a list of dicts with key = other paragraph number, values = theirs cossin score
    cossinDict = {} 
    for paragraph in paragrapFreqDict: 
        if paragraph not in cossinDict:
            cossinDict[paragraph]={}
        for otherParagraph in paragrapFreqDict:
            if otherParagraph!=paragraph:
                cosSinScore=cosSim(paragrapFreqDict[paragraph],paragrapFreqDict[otherParagraph])
                if cosSinScore >= scoreThreshold: #threshold check
                    cossinDict[paragraph].update({otherParagraph:cosSinScore})
    
    return cossinDict



def categorizedNumberOfTopic(groupedTopics):
    '''Put paragraph into group which share cosSim similarity
    
    Input: cosSin dictionary from groupedTopic()
    Output: dict contains topic ID as key and set of paragraph share the same similarity
    '''
    topicID = 0
    similarParagrapInGroup=[]
    
    #loop through grouped Topic and save all paragraph with similarity in a list
    for paragraph, relatives in groupedTopics.items(): 
        similarParagraph=[]
        similarParagraph.append(paragraph)
        for relative in relatives:
            similarParagraph.append(relative)
        similarParagrapInGroup.append(similarParagraph) #create list of list of similarity paragraph
    
    #merge lists which contain common paragraph to reduce topic number     
    topicsDict = {}
    uniqueParagraphGroup = []
    
    while len(similarParagrapInGroup)>0:
        first, *rest = similarParagrapInGroup #divide list in to first and rest sets 
        first = set(first)

        lf = -1
        #find all list with at least 1 common element then merge those together
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest: #for each list in rest
                if len(first.intersection(set(r)))>0: #when there is intersection between list and first list
                    first |= set(r) #reassigned first list with the list r 
                #when list with no intersect then add it in rest2 which contain only no intersect list 
                #for future iterator
                else:
                    rest2.append(r)   
        #save topic and its group of paragraph in a dict
        topicsDict[topicID]=first
        topicID+=1
        similarParagrapInGroup = rest2 #set original list to rest2 to start loop for the rest list which does not have intersec

    return topicsDict


createInput() #create input text

#lemmatized, filter input text and create token word 
filterTokenizedWord,lemmatizedText = tokenizeInput('newsgroups.txt')
#create word frequency vectors for paragraphs
wordFrequencyParas=frequencyTokenVector(filterTokenizedWord,lemmatizedText)
#group paragraph based on cosSim score, idea score 0.5 but sample tend to have lower score
groupedTopics = groupedTopic(wordFrequencyParas,0.2)
print('number of topic and paragraph which share similarity', '\n',categorizedNumberOfTopic(groupedTopics),'\n')

#extract topic of paragraph based on top 8 common word
topicDict=extractTopic(wordFrequencyParas, 8)
#Test the topic 
print('Check similarity with most comon word for para 73 and 92','\n',topicDict[73],'\n',topicDict[92],'\n')





