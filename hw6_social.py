"""
Social Media Analytics Project
Name:
Roll Number:
"""

# from pandas.io.parsers import count_empty_vals
import hw6_social_tests as test
import re

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    return pd.read_csv(filename)
    
   
    


'''

parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    start = fromString.find("From:") +  len("From:")
    fromString = fromString[start:]
    end = fromString.find(" (")
    fromString = fromString[:end]
    fromString = fromString.strip()
    name=fromString
    return name


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    start = fromString.find(" (") + len(" (")
    fromString = fromString[start:]
    end = fromString.find("from")
    fromString = fromString[:end]
    fromString = fromString.strip()
    position=fromString
    return position


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    start = fromString.find(" from") + len(" from")
    fromString = fromString[start:]
    end = fromString.find(")")
    fromString = fromString[:end]
    fromString = fromString.strip()
    state=fromString
    return state

    


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    word = message.split('#')
    hashtags = []
    temp_string = ""
    for i in range(1,len(word)):
        for j in word[i]:
            if j in endChars:
                break
            else:
                temp_string += j
        temp_string = "#" + temp_string
        hashtags.append(temp_string)
        temp_string = ""
    return hashtags




'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    region = stateDf.loc[stateDf['state'] == state, 'region']
    return (region.values[0])


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []
    for index, row in data.iterrows():
        #print(row)
        val = row["label"]
        name = parseName(val)
        position = parsePosition(val)
        state= parseState(val)
        region = getRegionFromState(stateDf,parseState(val))
        val2=row['text']
        hashtag = findHashtags(val2)
        names.append(name)
        positions.append(position)
        states.append(state)
        regions.append(region)
        hashtags.append(hashtag)
    data['name']= names
    data['position'] = positions
    data['state'] = states
    data['region'] = regions
    data['hashtags'] = hashtags 
    return None


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"
    


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments =[]
    for index, row in data.iterrows():
        val = row['text']
        sentiments.append(findSentiment(classifier, val))
    data['sentiment'] = sentiments
    return None


    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    datadict = {}
    for index, row in data.iterrows():
        if ((len(colName)==0 and len(dataToCount) == 0) or (row[colName] == dataToCount)) :
            state = row['state']
            if state  not in datadict:
                datadict[state] = 0
            datadict[state] += 1
    return datadict

    


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    regions = {}
    convert_dictionary_groupby = dict(data.groupby(["region", colName]).size())
    print(convert_dictionary_groupby)
    for key in convert_dictionary_groupby:
        region = key[0]
        inner_key = key[1]
        value = convert_dictionary_groupby[key]
        if region not in regions:
            regions[region] = {}
            regions[region][inner_key] = value
        else:
            regions[region][inner_key] = value
    print(regions)
    return regions 
    


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtag_count_dict = {}
    for index, row in data.iterrows():
        hashtags = row["hashtags"]
        for hashtag in hashtags:
            if hashtag not in hashtag_count_dict:
                hashtag_count_dict[hashtag] = 0
            hashtag_count_dict[hashtag] += 1
    # print(hashtag_count_dict)
    return hashtag_count_dict


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    most_common_hashtags_dict = {}
    for key, value in sorted(   hashtags.items(), key=lambda item: item[1], reverse=True):
        most_common_hashtags_dict[key]=value
        if len(most_common_hashtags_dict) == count:
            break
    return most_common_hashtags_dict

    


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    message_count = 0
    score_lst = []
    for index, row in data.iterrows():
        if hashtag in row['text']:
            message_count += 1
            if row['sentiment'] == 'positive':
                value = 1
                score_lst.append(value)
            if row['sentiment'] == 'negative':
                value = -1
                score_lst.append(value)
            if row['sentiment'] == 'neutral':
                value = 0
                score_lst.append(value)
    sentiment_score = sum(score_lst)/message_count
    return sentiment_score



### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    states = list(stateCounts.keys())
    count = list(stateCounts.values())
    fig = plt.figure(figsize = (10, 50))
    plt.bar(states, count, color ='maroon',
        width = 0.8)
    plt.xticks(rotation = 90)
    plt.title(title)
    plt.show()

    # state = []
    # count = []
    # for key, value in stateCounts.items():
    #     state.append(key)
    #     count.append(value)
    # plt.bar(state, count)
    
    # plt.xticks(rotation=90)
    # plt.show()
    # return None
    

'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    feature_rate_dict ={}
    top_states_dict ={}
    for i in stateFeatureCounts:
        feature_rate_dict[i] = (stateFeatureCounts[i] / stateCounts[i])
    for key, value in sorted(feature_rate_dict.items(), key=lambda item: item[1], reverse=True):
        top_states_dict[key]=value
        if len(top_states_dict) == n:
            break
    graphStateCounts(top_states_dict, title)
    
    
    
    


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()
    # df = makeDataFrame("data/politicaldata.csv")
    # stateDf = makeDataFrame("data/statemappings.csv")
    # addColumns(df, stateDf)
    # addSentimentColumn(df)
    # test.testGetHashtagSentiment(df)

    ## Uncomment these for Week 2 ##
    # print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    # test.week2Tests()
    # print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
