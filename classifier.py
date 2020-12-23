import pandas as pd
import joblib


class GenreClassifier:
    def __init__(self, clf_fpath, mlb_fpath, thresholds_fpath):
        self.pipeline = joblib.load(clf_fpath)
        self.multi_label_bin = joblib.load(mlb_fpath)
        self.thresholds = pd.read_csv(thresholds_fpath, index_col=0)

    def predict(self, sentence):
        genres_prob = self.pipeline.predict_proba([sentence])
        for n, yi in enumerate(self.multi_label_bin.classes_):
            genres_prob[:, n] = genres_prob[:, n] > self.thresholds.loc[yi, 'best_treshold']
        genres = self.multi_label_bin.inverse_transform(genres_prob)
        genres = list(zip(genres_prob, genres))
        genres = [g for _, g in sorted(genres, key=lambda x: x[0])]
        return genres
