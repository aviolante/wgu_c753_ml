import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from lib.parse_out_email_text import parse_out_text


# download stopwords (had to run 'Install Certificates.command')
nltk.download('stopwords')

# see stop words
stop_words = stopwords.words('english')
print(len(stop_words))  # 179

# stemming
stemmer = SnowballStemmer('english')

stemmer.stem('responsiveness')
# 'respons'

stemmer.stem('unresponsiveness')
# 'unrespons'

# mini project
ff = open("../ud120-projects/text_learning/test_email.txt", "r")
text = parse_out_text(ff)
print(text)

# original
# Hi Everyone!  If you can read this message, you're properly using parseOutText.
# Please proceed to the next part of the project!

# tokenized and stemmed
# ['hi', 'everyon', 'if', 'you', 'can', 'read', 'this', 'messag', 'your', 'proper', 'use', 'parseouttext', 'pleas',
#  'proceed', 'to', 'the', 'next', 'part', 'of', 'the', 'project']
