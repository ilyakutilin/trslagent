from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from iso639 import Lang


@pytest.fixture(scope="session")
def en_lang() -> Lang:
    return Lang("en")


@pytest.fixture(scope="session")
def ru_lang() -> Lang:
    return Lang("ru")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def sample_text() -> str:
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This is a second sentence about nothing in particular."
    )


@pytest.fixture
def sample_text_long() -> str:
    return (
        "Section 1: Introduction.\n\n"
        "This document describes the system architecture in detail. "
        "It covers the main components, their interactions, and the "
        "overall design philosophy.\n\n"
        "Section 2: Overview.\n\n"
        "The system consists of three primary modules. Each module "
        "is responsible for a specific domain of functionality."
    )


@pytest.fixture
def sample_xml_glossary() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<mtf>
<conceptGrp>
  <concept>1</concept>
  <languageGrp>
    <language lang="en" type="English"/>
    <termGrp>
      <term>flow meter</term>
    </termGrp>
  </languageGrp>
  <languageGrp>
    <language lang="ru" type="Russian"/>
    <termGrp>
      <term>расходомер</term>
    </termGrp>
  </languageGrp>
</conceptGrp>
<conceptGrp>
  <concept>2</concept>
  <languageGrp>
    <language lang="en" type="English"/>
    <termGrp>
      <term>pressure valve</term>
      <term>relief valve</term>
    </termGrp>
  </languageGrp>
  <languageGrp>
    <language lang="ru" type="Russian"/>
    <termGrp>
      <term>клапан давления</term>
      <term>предохранительный клапан</term>
    </termGrp>
  </languageGrp>
</conceptGrp>
</mtf>"""


@pytest.fixture
def sample_user_glossary_lines() -> list[str]:
    return [
        "flow meter = расходомер",
        "pressure valve = клапан давления",
        "# this is a comment",
        "actuator;drive = привод;исполнительный механизм",
    ]


@pytest.fixture
def mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.get_reply_async.return_value = ("Mocked translation text", "mock-completion-id")
    return mock
