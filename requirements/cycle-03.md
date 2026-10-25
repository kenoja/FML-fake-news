# Requirements for cycle 3

Drafted from the team's feedback on cycle 2 (collected 2026-10-25) by `scripts/collect_feedback.py`, then edited with the repo owner on 2026-10-25: decisions, resolved conflicts and the reviewer's open points from cycle 2 are added below. The agent treats every numbered item as a requirement; where an item and a decision below differ, **the decision wins**.

## Carried over

All requirements in earlier files (`requirements/cycle-01.md` ... `cycle-02.md`) still apply unless an item below changes them. Later items win over earlier ones.

## Decisions taken with the owner (25 Oct 2026)

- **D1. Merge results and error analysis (item 16).** Sections `08-results` and `09-where-they-fail` become one
  section, `08-results-and-why-they-fail`: the four models' results, then where and *why* they fail (examples,
  model errors vs label problems), as one connected analysis. Limitations of the study itself stay in a short
  separate section. New slugs (also in `AGENTS.md`):
  `08-results` + `09-where-they-fail` → `08-results-and-why-they-fail`; `10-limitations` → `09-limitations`;
  `11-conclusion` → `10-conclusion`; `12-how-we-used-ai` → `11-how-we-used-ai`. Delete the old files.
- **D2. Which model is which (items 5, 9).** `01-question-and-hypotheses` says once, plainly: analyses 1–3 are
  logistic regression on different inputs, analysis 4 is the LLM (no training). Later sections refer back to
  this instead of repeating it. The logistic regression code is still shown first in `04-word-counts`, because
  that is where the reader first meets it (Member A's reply to item 9).
- **D3. Hand check (items 2, 17).** The team does the 40-article check with a separate private tool during
  this cycle's feedback period; its results arrive for cycle 4. In this cycle: **do not change the 40 `doc_id`s
  in `results/tables/stance_check_sample.csv`** (the tool is built on them), and word the error-analysis text
  so that it states plainly which part rests on the automatic coding and that the team's hand check follows.
  No "unverified"/"unchecked" status language beyond that one statement.
- **D4. LLM prompts (item 14).** Keep all three prompt variants (the reply withdrew "only keep v1"). Do add the
  critical assessment of whether scoring the LLM against website labels is a fair comparison with the trained
  models.
- **D5. Executive summary vs introduction (item 6).** The report starts with a short executive summary (results
  allowed), followed by the introduction (no results). Both go in the `## Report` part of
  `01-question-and-hypotheses`, under `### Executive summary` and `### Introduction`.
- **D6. Hypotheses (items 5, 15).** Number them H1, H2, ... with no sub-letters (no H2a–c); each written out in
  full with its expectation, so it can be read without the others.
- **D7. Training environment (item 1, reply).** State where and how fast the models ran (machine and runtime
  from `results/run_info.json`) and what that means: everything runs on an ordinary laptop-class machine in
  minutes; only the LLM needs an internet connection and a paid API.
- **D8. References (items 2, 3).** Back design choices with classic machine-learning references (e.g. textbooks
  on logistic regression, regularisation, TF-IDF, evaluation, McNemar's test) and the 9/11 conspiracy
  background with published sources. Never invent a reference: if a source cannot be confirmed, leave the claim
  out.
- **D9. Item 16 wording.** The comment says "validation vs training set"; the comparison shown is validation vs
  **test**. Explain why scores are higher on validation than on test (different and fewer websites, one large
  site in test).

## Open points from the cycle-2 review (still to fix)

- **R1.** Remove Cohen's kappa everywhere it is still explained or coded (section 09, glossary,
  `src/fml/errors.py`): requirement 13 of cycle 2 removed it. (The cycle-4 agreement measure is decided then.)
- **R2.** No test-set numbers in the method sections `04`–`07` (section 06 still quotes test F1 and a p-value).
- **R3.** The error table in the report shows raw column names (`lr_only_wrong`, `stance_no_claim`,
  `not_sampled`): use plain labels a non-technical reader understands.
- **R4.** Remove process remarks from report text: the repeated "split file was made before cleaning"
  paragraph and the per_split/metrics.json remark in `03`, "None of them is recorded as done", and similar.
- **R5.** The licence status stays in `docs/decisions.md` only; the site and report only say where the data
  come from and that the texts are not redistributed (cycle-2 requirement 17).
- **R6.** Code shown in cells must meet the beginner style (cycle-2 requirement 14), including `mattr` and
  `style_row` in `06`; remove unused code.
- **R7.** Resolve the open items in `docs/decisions.md` "Needs a team decision" that are not team decisions
  (e.g. report length counting, near-duplicate choice): decide conservatively, state the choice once in the
  right section, and close them.
- **R8.** Report length: stay well inside 3,000–5,000 words (aim for about 4,500) so later edits fit.

## Sign-off status

| section | Member A | Member B | everyone |
|---|---|---|---|
| 00-how-to-use | – | 👍 | no |
| 01-question-and-hypotheses | – | – | no |
| 02-data-and-cleaning | – | 👍 | no |
| 03-split-by-website | – | 👍 | no |
| 04-word-counts | – | – | no |
| 05-tf-idf | – | – | no |
| 06-feature-engineering | – | 👍 | no |
| 07-llm | – | – | no |
| 08-results | – | – | no |
| 09-where-they-fail | – | – | no |
| 10-limitations | – | 👍 | no |
| 11-conclusion | – | 👍 | no |
| 12-how-we-used-ai | – | – | no |

**Cycle finished for everyone:** no

## Feedback by section

For **Unclear** items: answer the question in that section of the walkthrough (and the report, if relevant). For **Change** items: make the change, or record in `docs/decisions.md` why not. **Other** items: use judgement; record what you did.

### 00-how-to-use

1. **[Other]** (Member A) - general: better explanations for the code
- clear justification for design choices
- adding a touch of human writing in terms of story telling and flow for the text. the sections should flow smoothly into each other
   - reply (Member A): - adding runtime, the server and implications on which the trainig of the models was performed on
   - reply (Member A): - the code explanations should follow the structure that not only one comment explains the whole code, but just a short phrase what each code snippet does
2. **[Other]** (Member A) - the by hand tests were not made this run yet, will be done on the next one
- find references for design choices. try to stick to classics for ML as this is a very classic ML assingment

### 01-question-and-hypotheses

3. **[Other]** (Member B) The first sentence in the report text is very unclear. This first sentence needs to be changed to a brief introduction namely, "9/11 is one of the oldest and most persistent topics in online conspiracy content ("inside job", "controlled demolition", "false flag"). Platforms and fact-checkers increasingly rely on automatic filters to flag such content". Next, in th sentence, "This study asks whether the text of a full article about the 9/11 attacks reveals if it was published on a conspiracy-oriented or on a mainstream website, and how four methods compare on this task", simply saying four methods is not specific enough, it needs to be changed to machiene learning methods for text recognition.
   - reply (Member A): Adding references for the suggested sentences needed.
4. **[Other]** (Member B) I am a little confused about the following sentence "The hypotheses were chosen from the literature and fixed before the final test run; the exact date on which they were written down cannot be shown." - i think we should remove the latter part of the sentence.
   - reply (Member A): Agreed.
5. **[Other]** (Member B) I think the four methods that we are using should be explained a little bit better, we need to state what we expect out of them. Also we need to justify better why we explicitly chose these 4 methods. After this we go into the hypothesis. 

- What we expected for the Word Counts: We expect plain word counts to tell the two kinds of website apart better than always guessing mainstream, because the two kinds tend to use different words.

- What we expected for the TF-IDF: Giving rare, telling words more weight than everyday words improves on plain word counts.

- What we expected for the TF-IDF plus extra features: Adding features about certainty and writing style improves the TF-IDF model.

- What we expected for the LLM classifier: We expect a cheap language model with no training to do roughly as well as the trained models, because it has read a great deal of text and can judge how an article is written.

In a later explanation the following was written : "We used logistic regression for every trained model because it is simple and its weights can be read as word lists. We did not try other classifiers, because the question is about features, not which method is best. We did not fine-tune a language model, which needs more data and money than we have. We left out emotion features, because fear and anger are normal in reports on an attack and so tell us little." I think this could be used nicely in the explanation of the 4 methods.
6. **[Other]** (Member A) Clear differentiation between executive summary and introduction are needed. no results in the introduction.

### 02-data-and-cleaning

7. **[Other]** (Member A) - adding justification for cleaning choices. 
- risks for missed labels and website style are missing: model could be using shortcut learning and therefore have target leakage problems
- importance of this step should be clarified!!

### 03-split-by-website

8. **[Other]** (Member A) - explanation why removing website labels and so on are relevant to be removed anyways even when splitting by article is still needed is missing

### 04-word-counts

9. **[Other]** (Member B) If we are mentioning in the beginning section that we are using logistic regression for all 4 models then we do not need to mention it here. 

We should probably refer back to our expectation of this model that we are making in the beginning section. 

The last sentence, of the first paragraph, in this section says "The test set was not used." - this needs to be reviewed
   - reply (Member A): - I disagree on the log regression, as this is the first part where the technical part in form of code is shown. if this is not wanted here then the code for the log regression has to go somewhere else in the report
10. **[Other]** (Member A) - difference between scores not clearly explained and which is the relevant factor for the choice of C
- choose clearer variable name for "C"
- word list should be two separate tables, not under each other

- explanation on which choices were not made only for ML specific (mainly log regression) not for LLM etc
   - reply (Member A): score differences refers to "precision" "recall" and "F1". what has which importance and how it can be read

### 05-tf-idf

11. **[Other]** (Member B) I like this section, the only thing I would add is a few sentences that reflect on the expectation made in the beginning section.

### 06-feature-engineering

12. **[Other]** (Member A) - ablation is not clear
- mattr is also not clearly explained why and how it is done. standardizing the vectors to standardize lenght was also not stated
13. **[Other]** (Member A) - analysis why all TF IDF plus three are worse than just TF IDF

### 07-llm

14. **[Other]** (Member A) - simplification: only keep v1. remove all other prompt suggestions. ciritically assess why this is the best way to score the LLM vs the model
   - reply (Member A): nvm, lets keep all three

### 08-results

15. **[Other]** (Member B) I like it the only thing I would change is write out every hypothesis, this means that H2a, H2b, H2c should have a little more context.
16. **[Other]** (Member A) - combine the results and limitations with where and especially why they fail into one analysis
- make sure it is clear why the confusion matrixes look so much better on validation set than trainings set
   - reply (Member A): - justify the choice for mcneemar

### 09-where-they-fail

17. **[Other]** (Member B) Okay this section makes sense besides the ending. The last paragraph says this: "The automatic stance coding is not verified: no person has checked it yet, and the team will check 40 articles by hand. The label-problem share (47 of 65 readable codings) is therefore an estimate from one unchecked source." - I think we should revise this
   - reply (Member A): we have to manually check the set to identify this. otherwise the section does not make sense. not done yet, therefore the wording

### 12-how-we-used-ai

18. **[Other]** (Member A) - explain the claude code setup with the agents, the website and multiple iterations -  not which choices were made by claude

## Conflicts to decide

- Item 9 (do not repeat "logistic regression" in 04) vs Member A's reply (the code is first shown there):
  resolved by D2.
- Item 14 ("only keep v1") vs its reply ("keep all three"): resolved by D4, keep all three.
- Items 2/17 (the hand check is needed for the error analysis to make sense) vs running this cycle before it
  is done: resolved by D3.
