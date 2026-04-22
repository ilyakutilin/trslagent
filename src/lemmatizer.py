import re

import pymorphy3
import spacy
from iso639 import Lang


class Lemmatizer:
    def __init__(self, phrase: str, lang: Lang) -> None:
        self.phrase = phrase
        self.lang = lang

    def _lemmatize_english(self) -> str:
        nlp = spacy.load(
            "en_core_web_sm", disable=["ner", "parser"]
        )  # only need tagger for lemmas

        # Normalize hyphens before lemmatizing
        normalized = self.phrase.replace("-", " ").replace("–", " ")

        # Lemmatize a multi-word phrase
        doc = nlp(normalized)
        lemmas = [
            token.lemma_.lower() for token in doc if token.is_alpha or token.is_digit
        ]
        return " ".join(lemmas)

    def _lemmatize_russian(self):
        words = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", self.phrase)
        morph = pymorphy3.MorphAnalyzer()
        lemmas = [morph.parse(word)[0].normal_form for word in words]
        return " ".join(lemmas)

    def lemmatize(self) -> str | None:
        def unsupported():
            return None

        lemmatizers = {
            Lang("en"): self._lemmatize_english,
            Lang("ru"): self._lemmatize_russian,
        }

        return lemmatizers.get(self.lang, unsupported)()
