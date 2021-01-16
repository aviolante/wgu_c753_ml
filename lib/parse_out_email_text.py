from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer


def parse_out_text(f):

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(str.maketrans("", ""))

        # project part 2: comment out the line below
        words = text_string

        # tokenizer words
        word_tokens = word_tokenize(words)

        # stem tokenized
        stemmer = SnowballStemmer('english')

        stemmed_words = []

        for word in word_tokens:
            stemmed_words.append(stemmer.stem(word))

    return stemmed_words


