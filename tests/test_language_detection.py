from __future__ import annotations

import pytest
from iso639 import Lang

from src.config import InputData, Settings
from src.language_detection import (
    detect_language,
    resolve_languages,
)


class TestDetectLanguage:
    def test_detect_english(self):
        result = detect_language(
            "This is a simple English text used for translation. "
            "It contains multiple sentences so that the detector "
            "has enough context to work with."
        )
        assert result == Lang("en")

    def test_detect_russian(self):
        result = detect_language(
            "Это простой русский текст, используемый для перевода. "
            "Он содержит несколько предложений, чтобы детектор "
            "имел достаточно контекста для работы."
        )
        assert result == Lang("ru")

    def test_detect_french(self):
        result = detect_language(
            "Ceci est un texte français simple utilisé pour la traduction. "
            "Il contient plusieurs phrases pour que le détecteur "
            "ait suffisamment de contexte pour travailler."
        )
        assert result == Lang("fr")

    def test_detect_german(self):
        result = detect_language(
            "Dies ist ein einfacher deutscher Text für die Übersetzung. "
            "Er enthält mehrere Sätze, damit der Detektor "
            "genügend Kontext zum Arbeiten hat."
        )
        assert result == Lang("de")

    def test_detect_with_max_chars(self):
        long_text = (
            "This is an English document that needs to be translated. "
            "The system architecture is described in great detail. "
        ) * 20
        result = detect_language(long_text, max_chars=100)
        assert result == Lang("en")

    def test_uses_only_first_n_chars(self, mocker):
        from src import language_detection

        mock_detect = mocker.patch.object(language_detection, "detect")
        mock_detect.return_value = "en"

        text = "Hello world. " * 100
        detect_language(text, max_chars=100)
        assert len(mock_detect.call_args[0][0]) == 100

    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError, match="no detectable features"):
            detect_language("")

    def test_short_gibberish_falls_through(self):
        result = detect_language("a" * 200)
        assert isinstance(result, Lang)

    def test_unknown_code_raises(self, mocker):
        from src import language_detection

        mocker.patch.object(language_detection, "detect", return_value="zz")

        with pytest.raises(ValueError, match="does not recognise"):
            detect_language("some text to get past length checks")

    def test_detect_spanish(self):
        result = detect_language(
            "Este es un texto español simple utilizado para la traducción. "
            "Contiene múltiples oraciones para que el detector "
            "tenga suficiente contexto."
        )
        assert result == Lang("es")

    def test_detect_returns_lang_instance(self):
        result = detect_language("Hello world. This is a test.")
        assert isinstance(result, Lang)
        assert result.pt1 == "en"

    def test_max_chars_default(self):
        result = detect_language("Hello world. " * 100)
        assert result == Lang("en")


class TestResolveLanguagesTranslation:
    def _make_cfg(
        self,
        source_lang: Lang | None,
        target_lang: Lang | None,
        source_text: str,
        target_text: str | None = None,
    ) -> Settings:
        return Settings(
            input_data=InputData(
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source_text,
                target_text=target_text,
            ),
        )

    def test_both_none_english_source(self):
        """Translation mode: source=English, both None -> source=en, target=ru."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=None,
            source_text=(
                "This is an English document that needs to be translated. "
                "The document describes the system architecture."
            ),
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    def test_both_none_russian_source(self):
        """Translation mode: source=Russian, both None -> source=ru, target=en."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=None,
            source_text=(
                "Это русский документ, который нужно перевести. "
                "Документ описывает архитектуру системы."
            ),
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("ru")
        assert cfg.input_data.target_lang == Lang("en")

    def test_source_set_target_none_en(self):
        """Source explicit 'en', target None -> target=ru."""
        cfg = self._make_cfg(
            source_lang=Lang("en"),
            target_lang=None,
            source_text="Some English text to translate.",
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    def test_source_set_target_none_ru(self):
        """Source explicit 'ru', target None -> target=en."""
        cfg = self._make_cfg(
            source_lang=Lang("ru"),
            target_lang=None,
            source_text="Русский текст для перевода.",
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("ru")
        assert cfg.input_data.target_lang == Lang("en")

    def test_both_explicit_no_detection(self, mocker):
        """Both explicit -> no detection."""
        from src import language_detection

        mock_detect = mocker.patch.object(language_detection, "detect")
        mock_detect.return_value = "en"

        cfg = self._make_cfg(
            source_lang=Lang("fr"),
            target_lang=Lang("de"),
            source_text="Texte français.",
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("fr")
        assert cfg.input_data.target_lang == Lang("de")
        mock_detect.assert_not_called()

    def test_source_none_target_explicit(self):
        """Source None, target explicit -> detect source only."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=Lang("ru"),
            source_text="This is English text.",
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    def test_same_language_raises(self):
        """Same detected source and target -> ValueError."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=None,
            source_text=(
                "This is an English document. It will be allegedly "
                "translated but same lang should raise."
            ),
            target_text=(
                "This is another English text. For review mode this "
                "should raise because sources match."
            ),
        )
        with pytest.raises(
            ValueError, match="Source and target languages are the same"
        ):
            resolve_languages(cfg)

    def test_target_none_french_source_defaults_to_ru(self):
        """Source is fr (not ru) -> target defaults to ru."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=None,
            source_text=(
                "Ce document doit être traduit. Il décrit l'architecture du système."
            ),
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("fr")
        assert cfg.input_data.target_lang == Lang("ru")


class TestResolveLanguagesReview:
    def _make_cfg(
        self,
        source_lang: Lang | None,
        target_lang: Lang | None,
        source_text: str,
        target_text: str,
    ) -> Settings:
        return Settings(
            input_data=InputData(
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source_text,
                target_text=target_text,
            ),
        )

    def test_both_none_review_both_detected(self):
        """Review mode: both None -> detect both from texts."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=None,
            source_text=(
                "This is an English source document that needs "
                "to be reviewed against its translation."
            ),
            target_text=(
                "Это перевод на русский язык, который нужно проверить "
                "на соответствие исходному тексту."
            ),
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    def test_review_source_set_target_detected(self):
        """Review mode: source explicit, target None -> detect target."""
        cfg = self._make_cfg(
            source_lang=Lang("en"),
            target_lang=None,
            source_text="English source document.",
            target_text="Русский перевод документа.",
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    def test_review_target_set_source_detected(self):
        """Review mode: target explicit, source None -> detect source."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=Lang("ru"),
            source_text="English source document for review.",
            target_text="Русский перевод для проверки.",
        )
        resolve_languages(cfg)
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    def test_review_same_detected_languages_raises(self):
        """Review mode: both detected as same language -> ValueError."""
        cfg = self._make_cfg(
            source_lang=None,
            target_lang=None,
            source_text="This is English text for review.",
            target_text="This is also English text pretending to be a translation.",
        )
        with pytest.raises(
            ValueError, match="Source and target languages are the same"
        ):
            resolve_languages(cfg)


class TestDetectLanguageIntegration:
    """Integration-style tests with the full Settings stack."""

    def test_source_lang_none_in_input_data(self, tmp_path):
        fp = tmp_path / "source.txt"
        fp.write_text("English source document for translation.")
        data = InputData(
            source_lang=None,
            target_lang=Lang("ru"),
            source_file_path=fp,
        )
        assert data.source_lang is None
        assert data.target_lang == Lang("ru")

    def test_target_lang_none_in_input_data(self, tmp_path):
        fp = tmp_path / "source.txt"
        fp.write_text("English source document.")
        data = InputData(
            source_lang=Lang("en"),
            target_lang=None,
            source_file_path=fp,
        )
        assert data.source_lang == Lang("en")
        assert data.target_lang is None

    def test_both_none_in_input_data(self, tmp_path):
        fp = tmp_path / "source.txt"
        fp.write_text("English source document.")
        data = InputData(
            source_lang=None,
            target_lang=None,
            source_file_path=fp,
        )
        assert data.source_lang is None
        assert data.target_lang is None
