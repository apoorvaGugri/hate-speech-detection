# Dataset card: Davidson et al. (2017), Hate Speech and Offensive Language

Checked on 2026-11-16 from the sources listed at the end. Figures marked **(reported)** come from
the literature or dataset cards and have not yet been re-computed here; `python -m src.data.eda`
computes them from the file itself and writes `reports/eda_summary.md`. Trust that file over this one.

## Source and licence

| | |
|---|---|
| Authors' repository | https://github.com/t-davidson/hate-speech-and-offensive-language |
| File used | `data/labeled_data.csv` (about 2.5 MB) |
| Hugging Face mirror | https://huggingface.co/datasets/tdavidson/hate_speech_offensive |
| Paper | Davidson, Warmsley, Macy, Weber. ICWSM 2017 |
| Repository status | The authors state it is no longer actively maintained |

**Licence: needs care.** The original Hugging Face loading script for this dataset declared the
licence as **MIT**, matching the GitHub repository. The current Hugging Face dataset card lists
the licence as **unknown**. Separately, the tweets were written by Twitter users and collected
under Twitter's terms, which an MIT licence on the repository does not override.

What this project does about it:

- downloads from the authors' repository at run time and never commits or redistributes the tweets;
- cites the paper;
- treats the data as research/educational use only.

Before any commercial use, open the `LICENSE` file in the GitHub repository yourself and take advice.

## Structure

One row per tweet, 24,783 rows **(reported; the download script checks this count)**, one split.

| Column | Meaning |
|---|---|
| (unnamed index) | Row id from the original file; dropped on load |
| `count` | Number of CrowdFlower annotators who labelled the tweet (minimum 3) |
| `hate_speech` | How many of them chose hate speech |
| `offensive_language` | How many chose offensive language |
| `neither` | How many chose neither |
| `class` | Majority label: `0` hate speech, `1` offensive language, `2` neither |
| `tweet` | Tweet text, with handles, URLs, `RT` markers and HTML entities |

The Hugging Face mirror names the vote columns `hate_speech_count`, `offensive_language_count`
and `neither_count`. `src/data/load.py` accepts both.

## Label distribution (reported)

| Class | Tweets | Share |
|---|---:|---:|
| 0 Hate speech | 1,430 | 5.8% |
| 1 Offensive language | 19,190 | 77.4% |
| 2 Neither | 4,163 | 16.8% |

Consequences for modelling: a classifier that always answers "offensive" reaches about 77%
accuracy. Accuracy is therefore not used for model selection; macro-F1 and per-class recall are.

## How it was built

Tweets were found by searching Twitter for terms from the Hatebase.org lexicon, the timelines of
the matching users were collected, and a random sample of about 25k tweets was labelled by
CrowdFlower workers, at least three per tweet, with the majority vote as the label.

## Privacy concerns

- Tweets contain **@handles** of authors' contacts and sometimes names. `text_cleaner` masks every
  handle and URL before text reaches a model or a report.
- Authors did not consent to research use, and some tweets have since been deleted. The raw file,
  predictions and error reports that quote tweets are git-ignored.
- Do not try to re-identify authors or look tweets up.

## Data-quality concerns

- **Label noise.** Majority vote over three annotators; the hate/offensive border is where they
  disagree most. The per-tweet vote columns let you measure this (EDA reports agreement).
- **Duplicates.** Retweets repeat text. Splitting without de-duplication leaks test data into training.
- **Markup noise.** HTML entities (`&amp;`, numeric emoji codes), `RT` prefixes, truncated tweets.
- **Short texts.** Most tweets are under 30 words, so there is little context per example.

## Limitations

1. **Lexicon sampling.** Because collection started from slur lists, almost every tweet contains
   strong language. Models learn "slur present" shortcuts and are weak on implicit hate.
2. **Racial and dialect bias.** The authors' follow-up work (Davidson, Bhattacharya, Weber, 2019)
   and Sap et al. (2019) found that classifiers trained on this data label tweets in
   African-American English as abusive at higher rates.
3. **Subjective, US-centric definitions.** Hate speech is defined as language expressing hatred
   toward a targeted group or meant to demean its members; annotators applied that with little context.
4. **Small minority class.** Roughly 1.4k hate-speech tweets; the test split holds about 140, so
   hate-speech metrics have wide confidence intervals.
5. **Time and platform.** English Twitter, collected before 2017. Slang and platform norms have moved.
6. **No conversation context.** Single tweets without threads, quoted posts or author information.

## Sources

- GitHub repository README (maintenance note, content warning, file layout).
- Hugging Face dataset cards `tdavidson/hate_speech_offensive` and the legacy `hate_offensive`
  (schema, label mapping, row count, licence fields).
- Röttger et al. (2021), *HateCheck*, appendix D.1 (row count, hate-speech definition, reported
  annotator agreement).
