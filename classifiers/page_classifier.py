from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression,
    mutual_info_classif,
    chi2,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer
from typing import List, Tuple
import typing
import re
import pandas as pd
import numpy as np
import pprint
import time


class TextBasedPageFeatureExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, page_numbers: bool = False):
        self._vectorizer = DictVectorizer()
        self.page_numbers = page_numbers

    def _to_feature_dictionary(
        self, datum: str
    ) -> typing.Dict[str, typing.Union[int, float, bool]]:
        feature_dict = {
            # These features obscure subsections and give a unifying section feature
            "00ddd": self.check_section_num(section="00", length=5, datum=datum),
            "00dddd": self.check_section_num(section="00", length=6, datum=datum),
            "01ddd": self.check_section_num(section="01", length=5, datum=datum),
            "01dddd": self.check_section_num(section="01", length=6, datum=datum),
            "02ddd": self.check_section_num(section="02", length=5, datum=datum),
            "02dddd": self.check_section_num(section="02", length=6, datum=datum),
            "03ddd": self.check_section_num(section="03", length=5, datum=datum),
            "03dddd": self.check_section_num(section="03", length=6, datum=datum),
            "04ddd": self.check_section_num(section="04", length=5, datum=datum),
            "04dddd": self.check_section_num(section="04", length=6, datum=datum),
            "05ddd": self.check_section_num(section="05", length=5, datum=datum),
            "05dddd": self.check_section_num(section="05", length=6, datum=datum),
            "06ddd": self.check_section_num(section="06", length=5, datum=datum),
            "06dddd": self.check_section_num(section="06", length=6, datum=datum),
            "007ddd": self.check_section_num(section="07", length=5, datum=datum),
            "07dddd": self.check_section_num(section="07", length=6, datum=datum),
            "08ddd": self.check_section_num(section="08", length=5, datum=datum),
            "08dddd": self.check_section_num(section="08", length=6, datum=datum),
            "09ddd": self.check_section_num(section="09", length=5, datum=datum),
            "09dddd": self.check_section_num(section="09", length=6, datum=datum),
            "10ddd": self.check_section_num(section="10", length=5, datum=datum),
            "10dddd": self.check_section_num(section="10", length=6, datum=datum),
            "11ddd": self.check_section_num(section="11", length=5, datum=datum),
            "11dddd": self.check_section_num(section="11", length=6, datum=datum),
            "12ddd": self.check_section_num(section="12", length=5, datum=datum),
            "12dddd": self.check_section_num(section="12", length=6, datum=datum),
            "13ddd": self.check_section_num(section="13", length=5, datum=datum),
            "13dddd": self.check_section_num(section="13", length=6, datum=datum),
            "14ddd": self.check_section_num(section="14", length=5, datum=datum),
            "14dddd": self.check_section_num(section="14", length=6, datum=datum),
            "15ddd": self.check_section_num(section="15", length=5, datum=datum),
            "15dddd": self.check_section_num(section="15", length=6, datum=datum),
            "16ddd": self.check_section_num(section="16", length=5, datum=datum),
            "16dddd": self.check_section_num(section="16", length=6, datum=datum),
            "17ddd": self.check_section_num(section="17", length=5, datum=datum),
            "17dddd": self.check_section_num(section="17", length=6, datum=datum),
            "21dddd": self.check_section_num(section="21", length=6, datum=datum),
            "22dddd": self.check_section_num(section="22", length=6, datum=datum),
            "23dddd": self.check_section_num(section="23", length=6, datum=datum),
            "25dddd": self.check_section_num(section="25", length=6, datum=datum),
            "26dddd": self.check_section_num(section="26", length=6, datum=datum),
            "27dddd": self.check_section_num(section="27", length=6, datum=datum),
            "28dddd": self.check_section_num(section="28", length=6, datum=datum),
            "31dddd": self.check_section_num(section="31", length=6, datum=datum),
            "32dddd": self.check_section_num(section="32", length=6, datum=datum),
            "33dddd": self.check_section_num(section="33", length=6, datum=datum),
            "34dddd": self.check_section_num(section="34", length=6, datum=datum),
            "35dddd": self.check_section_num(section="35", length=6, datum=datum),
            "40dddd": self.check_section_num(section="40", length=6, datum=datum),
            "41dddd": self.check_section_num(section="41", length=6, datum=datum),
            "42dddd": self.check_section_num(section="42", length=6, datum=datum),
            "43dddd": self.check_section_num(section="43", length=6, datum=datum),
            "44dddd": self.check_section_num(section="44", length=6, datum=datum),
            "45dddd": self.check_section_num(section="45", length=6, datum=datum),
            "46dddd": self.check_section_num(section="46", length=6, datum=datum),
            "47dddd": self.check_section_num(section="47", length=6, datum=datum),
            "table_of_contents": self.check_table_of_contents(datum),
            "page_length == 0": self.check_page_length(datum=datum, bin=(0, 0)),
            "page_length 1 : 100": self.check_page_length(datum=datum, bin=(1, 100)),
            "page_length 101 : 200": self.check_page_length(
                datum=datum, bin=(101, 200)
            ),
            "page_length 201 : 400": self.check_page_length(
                datum=datum, bin=(201, 400)
            ),
            "page_length 401 : 600": self.check_page_length(
                datum=datum, bin=(401, 600)
            ),
            "page_length 601 : 800": self.check_page_length(
                datum=datum, bin=(601, 800)
            ),
            "page_length 901 : 1_000": self.check_page_length(
                datum=datum, bin=(801, 1_000)
            ),
            "page_length 1_001 : 1_200": self.check_page_length(
                datum=datum, bin=(1_001, 1_200)
            ),
            "page_length 1_201 : 1_400": self.check_page_length(
                datum=datum, bin=(1_201, 1_400)
            ),
            "page_length 1_401 : 1_600": self.check_page_length(
                datum=datum, bin=(1_401, 1_600)
            ),
            "page_length 1_601 : 1_800": self.check_page_length(
                datum=datum, bin=(1_601, 1_800)
            ),
            "page_length 1_801 : 2_000": self.check_page_length(
                datum=datum, bin=(1_801, 2_000)
            ),
            "page_length 2_001 : 2_200": self.check_page_length(
                datum=datum, bin=(2_001, 2_200)
            ),
            "page_length 2_201 : 2_400": self.check_page_length(
                datum=datum, bin=(2_201, 2_400)
            ),
            "page_length 2_401 : 2_600": self.check_page_length(
                datum=datum, bin=(2_401, 2_600)
            ),
            "page_length 2_601 : 2_800": self.check_page_length(
                datum=datum, bin=(2_601, 2_800)
            ),
            "page_length 2_801 : 3_000": self.check_page_length(
                datum=datum, bin=(2_801, 3_000)
            ),
            "page_length 3_001 : 3_200": self.check_page_length(
                datum=datum, bin=(3_001, 3_200)
            ),
            "page_length 3_201 : 3_400": self.check_page_length(
                datum=datum, bin=(3_201, 3_400)
            ),
            "page_length 3_401 : 3_600": self.check_page_length(
                datum=datum, bin=(3_401, 3_600)
            ),
            "page_length 3_601 : 3_800": self.check_page_length(
                datum=datum, bin=(3_601, 3_800)
            ),
            "page_length 3_801 : 4_000": self.check_page_length(
                datum=datum, bin=(3_801, 4_000)
            ),
            "page_length 4_000 : 10_000": self.check_page_length(
                datum=datum, bin=(4_001, 10_000)
            ),
            "addendum_header/footer": self.check_addendum(datum),
        }
        if self.page_numbers:
            page_dict = {
                "page == 1": self.check_page_num(target=1, datum=datum),
                "page <= 5": self.check_page_num(target=5, datum=datum),
                "page <= 10": self.check_page_num(target=10, datum=datum),
                "page <= 20": self.check_page_num(target=20, datum=datum),
                "page <= 30": self.check_page_num(target=30, datum=datum),
            }
            feature_dict.update(page_dict)
        return feature_dict

    @staticmethod
    def check_page_length(datum: str, bin: tuple[int, int]):
        if bin[0] == 0 & len(datum) == 0:
            return 1
        elif bin[0] <= len(datum) <= bin[1]:
            return 1
        return 0

    @staticmethod
    def get_header_and_footer(datum: str) -> str:
        """Returns the top and bottom fifth of a page of text, merged as one strings,
        or if the page contains very little text, returns the whole text"""
        if len(datum) < 1000:
            return datum
        else:
            fifth = len(datum) // 5
            footer = datum[-fifth:-1]
            header = datum[1:fifth]
            return footer + header

    @staticmethod
    def check_addendum(datum):
        header_footer = TextBasedPageFeatureExtractor.get_header_and_footer(datum)
        pattern = r"(?i)addend.*"
        match = re.search(pattern, header_footer)
        if match:
            return 1
        return 0

    @staticmethod
    def check_page_num(datum, target):
        if len(datum) < 20:
            match = re.search(r"PAGE (\d+)$", datum[-10:])
        else:
            match = re.search(r"PAGE (\d+)$", datum)
        if match:
            page_number = int(match.group(1))
        if target == 1:
            if page_number == 1:
                return 1
        elif page_number < target:
            return 1
        return 0

    @staticmethod
    def check_section_num(section: str, length: int, datum: str) -> int:
        """checks for a given section number in header or footer in page"""
        if length == 5:
            pattern = r"\b{}\s?\d\d\s?\d?(?=\s|-)\b".format(section)
        elif length == 6:
            pattern = r"\b{}\s?\d\d\s?\d\d(?=\s|-)\b".format(section)
        header_footer = TextBasedPageFeatureExtractor.get_header_and_footer(datum)
        match = re.search(pattern, header_footer)
        if match:
            return 1
        return 0

    @staticmethod
    def check_table_of_contents(datum: str) -> int:
        """Checks for table of contenst in header or footer of page"""
        pattern = r"(?i)table\sof\scontents|\btoc\b"
        header_footer = TextBasedPageFeatureExtractor.get_header_and_footer(datum)
        match = re.search(pattern, header_footer)
        if match:
            return 1
        return 0

    def transform(self, X):
        X_raw = [self._to_feature_dictionary(datum) for datum in X]
        return self._vectorizer.transform(X_raw)

    def fit(self, X, y=None):
        X_raw = [self._to_feature_dictionary(datum) for datum in X]
        self._vectorizer.fit(X_raw)
        return self

    def get_feature_names_out(self):
        return self._vectorizer.get_feature_names_out()


class TextBasedPageClassifier(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        custom_vectorizer=True,
        tf_idf_word_vectorizer=False,
        tf_idf_character_vectorizer=False,
        polynomial_features=False,
        n_feat_pairs=2,
        word_ngram_range=(1, 2),
        char_ngram_range=(1, 11),
        k=10_000,
        min_df=2,
        page_number_features=False,
        # ex k=10_000
        **params,
    ):
        self.custom_vectorizer = custom_vectorizer
        self.tf_idf_word_vectorizer = tf_idf_word_vectorizer
        self.tf_idf_character_vectorizer = tf_idf_character_vectorizer
        self.polynomial_features = polynomial_features
        self.n_feat_pairs = n_feat_pairs
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.k = k
        self.min_df = min_df
        self.page_number_features = page_number_features
        self.label_encoder = LabelEncoder()
        # see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn-pipeline-pipeline

        feature_union_list = []
        if custom_vectorizer:
            feature_union_list.append(
                (
                    "custom-features",
                    TextBasedPageFeatureExtractor(
                        page_numbers=self.page_number_features
                    ),
                )
            )
        if tf_idf_word_vectorizer:
            feature_union_list.append(
                (
                    "word-ngrams",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=word_ngram_range,
                        max_features=k,
                        min_df=min_df,
                    ),
                )
            )

        if tf_idf_character_vectorizer:
            feature_union_list.append(
                (
                    "char-ngrams",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=char_ngram_range,
                        max_features=k,
                        min_df=min_df,
                    ),
                )
            )

        if polynomial_features:
            # Create polynomial features for word n-grams and custom features
            poly_pipeline = Pipeline(
                [
                    ("feature_extraction", FeatureUnion(feature_union_list[0:2])),
                    # reduce our vocabulary based on MI
                    (
                        "feature combos",
                        PolynomialFeatures(degree=n_feat_pairs, include_bias=False),
                    ),
                ]
            )
            feature_extraction = [
                ("poly_pipeline", poly_pipeline),
                feature_union_list[2],
            ]
            self.pipeline = Pipeline(
                [
                    ("feature_extraction", FeatureUnion(feature_extraction)),
                    (
                        "feature_selection",
                        SelectKBest(score_func=chi2, k=10_000),
                    ),
                    ("clf", LogisticRegression(multi_class="ovr")),
                ]
            ).set_params(**params)

        else:
            self.pipeline = Pipeline(
                [
                    # combine word and character n-grams with custom features
                    ("feature_extraction", FeatureUnion(feature_union_list)),
                    # reduce our vocabulary based on MI
                    (
                        "feature_selection",
                        SelectKBest(score_func=chi2, k=10_000),
                    ),
                    ("clf", LogisticRegression(multi_class="ovr")),
                ]
            ).set_params(**params)

    def transform(self, X):
        return self.pipeline["feature_extraction"].transform(X)

    def fit(self, X, y):
        # fit our label encoder
        _y = self.label_encoder.fit_transform(y)
        # fit our pipeline
        self.pipeline.fit(X, _y)
        return self

    def partial_fit(self, X, y):
        # transform X
        _X = self.transform(X)
        # transform labels
        _y = self.label_encoder.fit_transform(y)
        self.pipeline.named_steps["clf"]._partial_fit(_X, _y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X, y, classes_of_interest: list[str] = None, output_dict=False):
        # save predictions across folds
        all_y_true = []
        all_y_pred = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # loop over folds
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.fit(X_train, y_train)

            y_pred = self.pipeline.predict(X_test)
            y_pred = self.label_encoder.inverse_transform(y_pred)

            print(
                classification_report(
                    y_true=y_test, y_pred=y_pred, labels=classes_of_interest
                )
            )
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        if classes_of_interest:
            report = classification_report(
                y_true=all_y_true,
                y_pred=all_y_pred,
                labels=classes_of_interest,
                output_dict=output_dict,
            )
        else:
            report = classification_report(
                y_true=all_y_true, y_pred=all_y_pred, output_dict=output_dict
            )
        print(report)
        return report

    # FIXME implement me
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

    def get_top_feats(self) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Takes a classifier and a vectorizer, returns the top k and bottom k features
        FIXME: In the case of OVR there will be a set of weights for each class, this only
        accesses one of those sets.
        """
        clf = self.pipeline.named_steps["clf"]
        union = self.pipeline.named_steps["feature_extraction"]
        feat_indices = np.argsort(clf.coef_)

        pos_indices = feat_indices[0, -100:]
        pos_feats = union.get_feature_names_out()[pos_indices]

        # get scores out of clf.coef_ with indices and zip into return
        pos_scores = clf.coef_[0, pos_indices]

        return pd.DataFrame({"Features": pos_feats, "Weights": pos_scores})


classes_of_interest = [
    "ELECTRICAL",
    "BIDDING AND CONTRACT DOCUMENTS",
    "ADDENDA",
    "GENERAL REQUIREMENTS",
    "COVER PAGE",
    "PROCESS INTEGRATION",
]


def time_fit():
    df = pd.read_csv(
        "/home/m/git/lum/wml/python/lum/wml/data/hand_corrected_no_sub_section_training_data.csv"
    )
    df = df.fillna("")
    df["CONCAT"] = df["TEXT"] + " PAGE " + df["PAGE_NO"].astype(str)
    X = df["CONCAT"]
    y = df["SECTION_LABEL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    args = {
        "char_ngram_range": (5, 6),
        "custom_vectorizer": True,
        "k": 5000,
        "min_df": 4,
        "polynomial_features": True,
        "tf_idf_character_vectorizer": True,
        "tf_idf_word_vectorizer": True,
        "word_ngram_range": (1, 2),
        "page_number_features": False,
    }

    page_clf = TextBasedPageClassifier(**args)
    fit_start = time.time()
    page_clf.fit(X=X_train, y=y_train)
    fit_time = time.time() - fit_start
    predict_start = time.time()
    predict_time = time.time() - predict_start

    train_pages = len(X_train)
    test_pages = len(y_test)

    print(
        f"Time to fit on {train_pages} pages {round(fit_time,2)} seconds.  Time to predict: {round(predict_time,2)} on {test_pages} pages."
    )


def main():
    df = pd.read_csv(
        "/home/m/git/lum/wml/python/lum/wml/data/hand_corrected_no_sub_section_training_data.csv"
    )
    df = df.fillna("")
    df.reset_index(drop=True, inplace=True)
    df["CONCAT"] = df["TEXT"] + " PAGE " + df["PAGE_NO"].astype(str)
    X = df["CONCAT"]
    y = df["SECTION_LABEL"]

    args = {
        "custom_vectorizer": True,
        "tf_idf_word_vectorizer": True,
        "polynomial_features": False,
        "word_ngram_range": (1, 2),
        "k": 5_000,
        "tf_idf_character_vectorizer": True,
        "char_ngram_range": (5, 6),
        "min_df": 4,
        "page_number_features": False,
    }
    with open(
        "/home/m/git/lum/wml/python/lum/wml/data/TextBasedPageClassifier_log.txt", "a"
    ) as logfile:
        page_clf = TextBasedPageClassifier(**args)
        report = page_clf.evaluate(X=X, y=y, classes_of_interest=classes_of_interest)
        logfile.write("\n")
        logfile.writelines(pprint.pformat(args))
        logfile.write("\n")
        logfile.writelines(report)
