from src.preprocessing.text_cleaner import full_clean, light_clean
from src.preprocessing.vocabulary import PAD_ID, UNK_ID, Vocabulary


def test_light_clean_masks_handles_and_urls():
    out = light_clean("RT @some_user: Look at this https://t.co/abc123 NOW")
    assert "@some_user" not in out and "t.co" not in out
    assert out == "@user : Look at this http NOW"


def test_light_clean_keeps_case_and_decodes_html():
    assert light_clean("Tom &amp; Jerry") == "Tom & Jerry"


def test_repeated_mentions_collapse():
    assert light_clean("@a @b @c hello") == "@user hello"


def test_full_clean_normalises():
    out = full_clean("RT @user1: Sooooo GOOD!!! #SunnyDay http://x.co &#128514;")
    assert out == "usermention soo good sunnyday urllink"


def test_full_clean_keeps_apostrophes_and_stopwords():
    assert full_clean("You don’t know") == "you don't know"


def test_cleaners_handle_empty_and_none():
    assert light_clean(None) == "" and full_clean("") == "" and full_clean("!!!") == ""


def test_rt_inside_word_is_not_removed():
    assert "ART" in light_clean("ART class")


def test_vocabulary_build_and_encode():
    vocab = Vocabulary.build(["red blue red", "red green"], min_freq=2)
    assert len(vocab) == 3                      # pad, unk, "red"
    assert vocab.encode("red blue", max_len=4) == [2, UNK_ID, PAD_ID, PAD_ID]
    assert vocab.encode("red red red", max_len=2) == [2, 2]


def test_vocabulary_max_size_and_roundtrip():
    vocab = Vocabulary.build(["a a a b b c"], min_freq=1, max_size=4)
    assert len(vocab) == 4
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.encode("a b c", 3) == vocab.encode("a b c", 3)
