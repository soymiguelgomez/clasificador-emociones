import pandas as pd
import pytest

from src.data_prep import check_no_leakage, clean_dataframe
from src.text_cleaning import clean_text
from src.tokenizer import WordTokenizer


def test_clean_text_matches_corpus_format():
    assert clean_text("I'm SO happy!!! :)") == "i m so happy"
    assert clean_text("I don’t feel well") == "i don t feel well"


def test_clean_text_removes_html_residue():
    assert clean_text("i feel crappy about my progress a href http www") == \
        "i feel crappy about my progress"
    assert clean_text("i feel stress amp sadness") == "i feel stress and sadness"


def test_clean_text_keeps_regular_words():
    # "a" y "alt" solo se eliminan dentro de los patrones de marcado.
    assert clean_text("i feel like a winner") == "i feel like a winner"


def test_tokenizer_padding_truncation_and_oov():
    tok = WordTokenizer.fit(["i feel happy", "i feel sad"], max_tokens=10, max_len=4)
    assert tok.vocab[:2] == ["<pad>", "<unk>"]
    ids = tok.encode(["i feel happy", "i feel unknownword", "i feel i feel i feel"])
    assert ids.shape == (3, 4)
    assert ids[0, 3] == 0                      # relleno al final
    assert ids[1, 2] == 1                      # palabra desconocida
    assert (ids[2] > 1).all()                  # truncado a max_len


def test_tokenizer_save_load_roundtrip(tmp_path):
    tok = WordTokenizer.fit(["i feel happy today"], max_tokens=10, max_len=5)
    tok.save(tmp_path / "tok.json")
    loaded = WordTokenizer.load(tmp_path / "tok.json")
    assert (loaded.encode(["i feel happy"]) == tok.encode(["i feel happy"])).all()


def test_clean_dataframe_drops_duplicates_and_conflicts():
    df = pd.DataFrame({
        "text": ["i feel happy", "i feel happy", "i feel odd", "i feel odd", "i feel sad"],
        "label": [1, 1, 4, 5, 0],
    })
    clean, stats = clean_dataframe(df)
    assert stats["exact_duplicates_removed"] == 1
    assert stats["conflicting_texts"] == 1
    assert sorted(clean["text"]) == ["i feel happy", "i feel sad"]


def test_check_no_leakage_detects_overlap():
    a = pd.DataFrame({"text": ["x", "y"]})
    b = pd.DataFrame({"text": ["y"]})
    c = pd.DataFrame({"text": ["z"]})
    with pytest.raises(AssertionError):
        check_no_leakage(a, b, c)
