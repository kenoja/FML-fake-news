# Requirements for cycle 3

Drafted from the team's feedback on cycle 2 (collected 2026-10-25) by `scripts/collect_feedback.py`. **Edit this file before merging**: delete items the team rejected, and resolve the conflicts at the end. The agent treats every numbered item as a requirement.

## Carried over

All requirements in earlier files (`requirements/cycle-01.md` ... `cycle-02.md`) still apply unless an item below changes them. Later items win over earlier ones.

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

List here any items that contradict each other, with the team's decision. The agent will not blend contradicting requests.

- none recorded yet
