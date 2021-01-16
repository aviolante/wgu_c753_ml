from lib.get_data import get_data
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def compute_fraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if poi_messages == 'NaN':
        poi_messages = 0
    else:
        poi_messages

    if all_messages == 'NaN':
        all_messages = 0
    else:
        all_messages

    try:
        fraction = poi_messages / all_messages
    except ZeroDivisionError:
        fraction = 0

    return fraction


data_dict = get_data()

submit_dict = {}
for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    print(fraction_from_poi)
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    print(fraction_to_poi)
    submit_dict[name] = {"from_poi_to_this_person": fraction_from_poi,
                         "from_this_person_to_poi": fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi


def submit_dict():
    return submit_dict

# {'METTS MARK': {'from_poi_to_this_person': 0.04708798017348203, 'from_this_person_to_poi': 0.034482758620689655},
#  'BAXTER JOHN C': {'from_poi_to_this_person': 0, 'from_this_person_to_poi': 0},
#  'ELLIOTT STEVEN': {'from_poi_to_this_person': 0, 'from_this_person_to_poi': 0},
#  'CORDES WILLIAM R': {'from_poi_to_this_person': 0.013089005235602094, 'from_this_person_to_poi': 0.0},
#  'HANNON KEVIN P': {'from_poi_to_this_person': 0.03062200956937799, 'from_this_person_to_poi': 0.65625},
#  'MORDAUNT KRISTINA M': {'from_poi_to_this_person': 0, 'from_this_person_to_poi': 0}


# mini project
words_file = "../ud120-projects/feature_selection/word_data.pkl"
authors_file = "../ud120-projects/feature_selection/email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

# predict test data
y_pred = clf.predict(features_test)

print(accuracy_score(labels_test, y_pred))  # 1.0 (overfit)

