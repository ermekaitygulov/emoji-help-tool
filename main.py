import yaml
import streamlit as st
from classifier import GenreClassifier
import emoji

DEFAULT_CLF_FPATH = './genres_pipeline.joblib'
DEFAULT_MLB_FPATH = './mlb.joblib'
DEFAULT_THRESHOLD_FPATH = './threshold.csv'
EMOJI_YAML_FPATH = './emoji_map.yaml'
AVAILABLE_EMOJI_LIST = [emoji for emoji, e_unicode in emoji.UNICODE_EMOJI.items()
                        if 'face' in e_unicode][:20]

with open(EMOJI_YAML_FPATH) as fin:
    DEFAULT_EMOJI_MAP = yaml.safe_load(fin)

for genre, raw_list in DEFAULT_EMOJI_MAP.items():
    emoji_list = [f":{e.replace(' ', '_')}:" for e in raw_list]
    emoji_list = [emoji.emojize(e) for e in emoji_list]
    DEFAULT_EMOJI_MAP[genre] = emoji_list
    for e in emoji_list:
        if e not in AVAILABLE_EMOJI_LIST:
            AVAILABLE_EMOJI_LIST.append(e)


def predict_emoji(clf_input, clf, emoji_map):
    genre_list = clf.predict(clf_input)[0]
    prediction = list()
    for g in genre_list:
        prediction.extend(emoji_map[g])
    return prediction


def choose_emoji():
    emoji_map = dict()
    for g, default_option in DEFAULT_EMOJI_MAP.items():
        options = st.sidebar.multiselect(
            g,
            AVAILABLE_EMOJI_LIST,
            default=default_option,
        )
        emoji_map[g] = options
    return emoji_map


if __name__ == '__main__':
    classifier = GenreClassifier(
        DEFAULT_CLF_FPATH,
        DEFAULT_MLB_FPATH,
        DEFAULT_THRESHOLD_FPATH,
    )
    st.title('Emoji help tool')
    st.markdown('''
    This is demo for emoji recommendation system. 
    For example, if you are a user it can save some time and recommend \N{face with tears of joy}
    when you type some joke to your friend. Or if you are creator of sticker packs user can see
    some of them in recommendation window.
    Actually system predicts genres and after that maps it to emoji.
    ''')

    chosen_emoji_map = choose_emoji()

    input_sentence = st.text_input(label='Type in some English words.', value='I really like you.')
    st.write('(genres) {}'.format(classifier.predict(input_sentence)))

    emoji_list = predict_emoji(input_sentence, classifier, chosen_emoji_map)
    option = st.selectbox('Choose emoji: ', emoji_list)
    st.write(f'{input_sentence} {option}')
