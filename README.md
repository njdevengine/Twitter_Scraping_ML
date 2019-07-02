
# Scraping Twitter for Corporate Espionage
## Using Webscraping, Facial Recognition and ML to gather Open Source Intelligence on Competing Corporations

![headline](headline.png "Twitter Project")


```python
#CREATE COLOR NUMBERS COLUMN

from IPython.display import Image, display
import urllib.request
from PIL import Image as img
import pandas as pd

df = pd.read_excel('twitter_followers_detailed.xlsx')
print('got df')
n = 0
for imageName in list(df["profile_image_url"]):
    try:
        urllib.request.urlretrieve(imageName, '/ml/out.png')
        colors = len(img.open("ml/out.png",mode="r").getcolors(maxcolors=9999999))
        df.at[n,"color_number"] = colors    
    except:
        df.at[n,"color_number"] = 0.01
    n+=1
    print(n)
```


```python
#Create Face Detection Column
import os
try:
    os.mkdir("ml")
except:
    print("folder already exists...")
    
import face_recognition

n = 0
for i in list(df["profile_image_url"]):
    try:
        urllib.request.urlretrieve(i, r'ml/out.png')
        detection = face_recognition.load_image_file("ml/out.png")
        if face_recognition.face_locations(detection):
            df.at[n,"face_detection"] = 1
        else:
            df.at[n,"face_detection"] = 0
    except:
        df.at[n,"face_detection"] = 999
    n+=1
```


```python
#Get rid of photos that did not connect and default photos
df = df[df.face_detection !=999.0]
df = df[df.color_number != 184]
```


```python
#URL Detect Column
array = []
for i in list(df["url"]):
    if str(i) != "nan":
        array.append(1)
    else:
        array.append(0)
df["url_detect"] = array
```

![headline](profile.png "profile")


```python
#output people labeled individuals to a csv/ then convert it to text for NLTK analysis
final = df

i_bios = final[final.label == "Individual"][['bio']]
i_bios.to_csv('i_bios.csv')

import csv
csv_file = (r'i_bios.csv')
txt_file = (r'i_bios.txt')
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
```


```python
#keywords function takes a filename, "grams" an integer meaning
#1-2- or 3 word n-grams and minimum count or appearances of the word or phrase

def keywords(file,grams,count):
    with open(str(file),'r',encoding="latin1") as myfile:
        my_string=myfile.read().replace('\n', '')

    string = ''.join(ch for ch in my_string if ch not in exclude)

    tokens = word_tokenize(string)
    text = nltk.Text(tokens)

    #array is the tuple of ngrams, array2 is the count of appearances, array1 is joined tuples, array 1 & 2 can be zipped into a dataframe
    array =[]
    array2 =[]
    bgs = nltk.ngrams(tokens,int(grams))
    fdist = nltk.FreqDist(bgs)
    for k,v in fdist.items():
        if v > int(count):
            array.append(k)
            array2.append(v)

    array1 = []
    for i in range(len(array)):
        x = ' '.join(map(str,array[i]))
        array1.append(x)

    df = pd.DataFrame({'phrase':array1,'count': array2}).sort_values(by="count",ascending=False)
    for i in list(df['phrase']):
        whitelist.append(i.lower())
    df.to_csv('output_'+str(grams)+'.csv')
```


```python
#import our dependancies, create empty whitelist array
#exclude includes ,$"@_[*|%)#+-<~^/;`=!:'&?}>({]\. (things to filter out)

import nltk
import string
from nltk import word_tokenize
import pandas as pd

whitelist = []
exclude = set(string.punctuation)
```


```python
#get 1grams, appearing more than 30 times,bigrams more than 20,
#and trigrams more than 10 times from the individuals bios
keywords('i_bios.txt',1,30)
keywords('i_bios.txt',2,20)
keywords('i_bios.txt',3,10)
```


```python
#remove stopwords from the individuals keywords array
from nltk.corpus import stopwords

new_whitelist = []

stop = list(set(stopwords.words('english')))
additional_stop = ["business"]
stop += additional_stop
safewords = ["I"]
stop = [i for i in stop if i not in safewords]

for i in whitelist:
    try:
        if (i.split(" ")[0] not in stop) & (i.split(" ")[1] not in stop):
            new_whitelist.append(i)
    except:
        if i not in stop:
            new_whitelist.append(i)
        pass
```


```python
#remove dupes, filter non-alpha keywords and save it to individual_keywords variable
new_whitelist = list(set(new_whitelist))

for i in range(len(new_whitelist)):
    try:
        if new_whitelist[i].split(" ")[0].isalpha() == False:
            new_whitelist.pop(i)
    except:
        pass
    
individual_keywords = new_whitelist
```

![headline](individual_keywords.png "Individual Keywords")


```python
#reset our variables and do all the same for business keywords
new_whitelist = []
whitelist = []
```


```python
b_bios = final[final.label == "Business"][["bio"]]
b_bios.to_csv('b_bios.csv')
csv_file = (r'b_bios.csv')
txt_file = (r'b_bios.txt')
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
```


```python
keywords('b_bios.txt',1,30)
keywords('b_bios.txt',2,20)
keywords('b_bios.txt',3,10)
```


```python
stop = list(set(stopwords.words('english')))
additional_stop = ["business","us"]
stop += additional_stop
safewords = ["I"]
stop = [i for i in stop if i not in safewords]

for i in whitelist:
    try:
        if (i.split(" ")[0] not in stop) & (i.split(" ")[1] not in stop):
            new_whitelist.append(i)
    except:
        if i not in stop:
            new_whitelist.append(i)
        pass
    
new_whitelist = list(set(new_whitelist))

# filter non-alpha stuff
for i in range(len(new_whitelist)):
    if new_whitelist[i].split(" ")[0].isalpha() == False:
        new_whitelist.pop(i)
        
business_keywords = new_whitelist
```

![headline](business_keywords.png "Individual Keywords")


```python
business_keywords
```


```python
individual_keywords
```


```python
#Create keyword counter columns for individual keywords, and business keywords
i_nums = []
for x in range(len(final)):
    test = final["bio"][x]
    n = 0
    try:
        for i in individual_keywords:
            if i in test:
                n+=1
        i_nums.append(n)
    except:
        i_nums.append(0)
```


```python
b_nums = []
for x in range(len(final)):
    test = final["bio"][x]
    n = 0
    try:
        for i in business_keywords:
            if i in test:
                n+=1
        b_nums.append(n)
    except:
        b_nums.append(0)
```


```python
final["b_key_count"] = b_nums
final["i_key_count"] = i_nums
```


```python
final[["bio","b_key_count","i_key_count"]].sort_values(by="b_key_count",ascending=False).head(40)
```


```python
final[["bio","b_key_count","i_key_count"]].sort_values(by="i_key_count",ascending=False).head(40)
```


```python
final.to_csv('final_output.csv',encoding="UTF-8")
```
