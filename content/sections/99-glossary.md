---
title: Glossary
part: Reference
hypothesis: Short plain definitions let every teammate follow the other sections without prior machine-learning knowledge.
---

Every technical word used on this site is explained here with an everyday example. The first use of each term in a section links to its entry.

### Ablation
Also: Ablation study, Ablations

Taking one group of features out, or adding one in, to see what changes. Like removing one ingredient from a recipe to learn what it contributes.

### Accuracy
The share of all articles a model labels correctly. It can mislead when one group is much larger: a model that always says "mainstream" is right for about three in four of our test articles, yet finds no conspiracy article. That is why we use F1 instead.

### Algorithm
A fixed list of steps that a computer follows, like a recipe. Logistic regression, for example, is an algorithm that learns word weights from examples.

### API
Also: application programming interface

A way for one program to send requests to another over the internet and get answers back. We sent each article to the language model through such a service, much like ordering by filling in a form instead of phoning.

### Bag of words

Treating an article as a pile of words and ignoring their order. Like tipping a letter's words into a bag: you can count them but no longer see the sentences. Our first two analyses work this way.

### Baseline

A simple method we compare other methods against. If a fancy model cannot beat the baseline, the extra effort did not help. Like walking as the baseline against which a bicycle is judged.

### Boilerplate
Text that a website repeats on many pages, such as menus, "share this" buttons or a newsletter banner. It says nothing about the article but can give away the website.

### Bootstrap
Also: Bootstrap resampling

Redrawing articles at random with replacement, many times, and recomputing the score each time. The spread shows how shaky the score is. Like asking what a score would be on other, similar test sets.

### C (regularisation strength)
Also: C, C value

The number that sets how strong the regularisation penalty is, with an odd twist: small C means a strong penalty and large C a weak one. Like a strictness dial where low C asks for short, simple answers. We tried 0.01, 0.1, 1, 10 and 100 and kept the best on validation.

### Cache
Also: Saved answers

A file that stores each answer the first time it is computed, so a rerun reads the file instead of paying again. Like writing a result in a notebook so you need not recalculate. It stores ids and codes, never article text.

### Class weight
Also: Balanced class weight

A setting that makes mistakes on the rarer class count more during training. "Balanced" weights each class by how rare it is. Like giving double points for finding the rarer kind of letter.

### Classifier
Also: Classifiers

A program that puts each item into one of a few groups. Ours decides whether an article comes from a conspiracy-oriented or a mainstream website. Like a mail sorter who looks at each letter and drops it in one of two boxes.

### Column
Also: columns

One kind of number that describes every article, like one column in a spreadsheet where each row is an article. Each word in the vocabulary is one column, and so is each extra feature.

### Confidence interval
Also: 95% confidence interval, interval, Confidence intervals, Intervals

A range around a score that shows how uncertain it is. A 95% interval is built so that it would cover the true value in about 95 of 100 repeats of the study. Like a forecast saying "20 to 24 degrees".

### Confusion matrix

A small table that counts the right and wrong calls: conspiracy called conspiracy, conspiracy called mainstream, and the same for mainstream. Like a scoreboard showing which mistakes happen where.

### Corpus
Also: Dataset

A big collection of texts used for a study. Ours is the LOCO corpus, a set of web articles from conspiracy and mainstream websites. Think of a library shelf that holds only the books we study. The texts stay in a private data repository.

### Cosine similarity

A number from 0 to 1 that says how alike two articles are, based on their word vectors. Near 1 means almost the same words in the same proportions. Like comparing two shopping baskets by what they contain.

### Epistemic marker
Also: Epistemic markers, Epistemic words, certainty and uncertainty words, certainty and doubt words, doubt words

Words that show how sure a writer is ("epistemic" means "about knowing"): "clearly" and "proves" for certainty, "maybe" and "allegedly" for doubt. We count them with a short hand-made list. Like noticing whether a friend says "I know" or "I guess".

### Exact test
Also: Exact tests

A significance test that works out the chance directly from counting every possible outcome, instead of using a rough formula. It stays reliable when the numbers are small. Like finding the odds of ten heads in twelve coin flips by listing all the ways it can happen.

### F1 score
Also: F1

A single number between 0 and 1 that combines precision and recall, high only if both are high. We report it for the conspiracy class. Like a grade that averages two exams but punishes a very bad one.

### False positive
Also: False negative, False alarm, False alarms, Miss

A false positive is a mainstream article wrongly called conspiracy; a false negative is a conspiracy article wrongly called mainstream. Like a smoke alarm ringing without fire, or staying silent during one.

### Feature
Also: Features

One measurable clue about an article that a model can use, such as how often "hoax" appears or how long the sentences are. Like noticing a letter's stamp, colour and handwriting before guessing who sent it.

### Fine-tuning
Also: Fine-tune

Training an existing language model further on our own labelled articles. We did not do this. Like sending a trained assistant on a course just for our task.

### Frozen

Fixed once and never changed, such as the split of articles. We keep it frozen so every model faces the same exam. Like sealing the exam paper before anyone studies.

### Generalise
Also: generalises, generalisation, carry over

To work well on new cases, not only on the ones a model learned from. A student who understood the method can solve new exercises; one who memorised the answers cannot.

### Grid search
Also: Grid

Trying a short list of settings, such as several values of C, and keeping the one that scores best on validation. If the best is at the edge of the grid, an even more extreme value might do better. Like testing oven temperatures one by one.

### Grouped split
Also: Split by website

Dividing articles so that all articles of one website land in the same part (train, validation or test). Like putting whole families, not single people, into the same team, so a model cannot recognise a site it has already met.

### Hand check
Also: Blind check, blind

People read a sample of articles themselves and give the same kind of answer the machine gave, without seeing the machine's answer or the website ("blind"). Comparing the answers shows how far the machine can be trusted. Like a teacher re-marking a few exams that an assistant graded, without looking at the assistant's marks first.

### Hash
Also: Fingerprint

A short code computed from some input that changes completely if the input changes at all. We use it to tell whether a saved answer still matches the prompt. Like a fingerprint for a document.

### Hedges and boosters
Also: Hedge, booster, Hedges, Boosters

Hedges are words that soften a claim ("perhaps", "seems"); boosters make it stronger ("obviously", "undeniable"). Together they form our certainty and uncertainty word lists. Like saying "I think" versus "I am sure".

### Hypothesis
Also: Hypotheses

A clear guess about what we will find, that the results can prove wrong. It is chosen before the final test run, from earlier studies, not from the results. Like predicting that a new recipe tastes better before anyone tastes it.

### JSON
A plain text format for storing data as named values, for example {"label": "mainstream"}. Our result files and the language model's answers use it.

### Label
Also: Class, Labels

The correct answer we know for each article: conspiracy-oriented or mainstream website. It comes from the website, not from reading the article. Like the name of the sender written on every letter in a practice pile.

### Label noise

Labels that do not fit the article they are attached to. A mainstream site may publish a conspiracy-style piece, and the label still says mainstream. Like a shelf labelled "cookbooks" holding one novel.

### Leakage
Also: Data leakage, target leakage, leak

When information from the test articles sneaks into training, so the scores look better than they are. Like a student who has seen the exam questions in advance.

### LLM
Also: Large language model, language model

A program trained on huge amounts of text that writes answers to written questions. Ours is Claude Haiku 5.5. Like a very well-read assistant who has never met our corpus.

### Logarithm
Also: Log, Logarithmic

A way to squash big numbers so that growth slows down: 1, 10 and 100 become 0, 1 and 2. We use it so a word repeated fifty times does not count fifty times more than a word used once. Like counting the digits of a number instead of its size.

### Logistic regression
Also: Logistic regressions, Regressions

A simple model that gives each feature a weight, adds up the weighted values for an article and turns the total into a probability. Like a judge adding points for and against, then stating how likely the case is.

### MATTR
Also: moving-average type-token ratio, type-token ratio

A measure of how much a text repeats itself: the share of different words in a window of 100 words, averaged over every such window along the text. Like asking, in every page of a book, how many words are new. Averaging over windows keeps long texts from looking repetitive just because they are long.

### McNemar test

A test that compares two classifiers on the same articles by looking only at those where exactly one is right. It tells whether the difference is bigger than chance. Like two students answering the same exam, comparing their disagreements.

### Multiple testing
Also: Correction for multiple tests

The more tests you run, the likelier one comes out "significant" by luck alone. A correction makes the bar stricter, for three tests by dividing 0.05 by 3. Like a coin: ten flips make a surprising run of heads much more likely than one flip does.

### N-gram
Also: Bigram, trigram, N-grams

A run of n words in a row. Single words are unigrams, pairs such as "inside job" are bigrams, triples are trigrams. Like reading a phrase instead of one word at a time, which keeps a little of the word order.

### Near-duplicate
Also: Duplicate, Near-duplicates

An article that is almost the same as another one, such as the same story copied with small changes. We remove them so the model is not tested on what it has already seen. Like two photocopies with one line changed.

### Overfitting

When a model memorises quirks of its training articles and then does worse on new ones. Like a student who learns last year's answers by heart instead of understanding the topic.

### P-value

A number between 0 and 1 that says how surprising our result would be if there were really no difference. A small p-value (under 0.05) means chance alone is an unlikely explanation. Like how odd ten heads in a row would seem.

### Per-website recall
Also: Recall per website

Recall worked out separately for each website: of the conspiracy articles one website published, the share a model found. Like checking a teacher's marks class by class, to see which class drags the average down.

### Pipeline
The full chain of steps that turns the raw articles into results: cleaning, splitting, building features, training, scoring and drawing the tables. One command (fml run-all) runs all of it.

### Precision

Of the articles the model called conspiracy, the share that really come from conspiracy websites. Like a metal detector: of all beeps, how many were really metal.

### Probability

A number from 0 to 1 saying how likely something is. A model output of 0.9 means it is fairly sure the article is from a conspiracy website. We call it conspiracy above 0.5. Like a weather forecast's chance of rain.

### Prompt
Also: Prompt variant, Prompts

The written instruction we give a language model, including the question and the article. A different wording is a prompt variant. Like the exact question on a worksheet: small changes can change the answers.

### Random seed
Also: Seed

A starting number that makes a computer's random choices repeatable. With the same seed, the same split or the same shuffle happens again. Like fixing the order of a shuffled deck.

### Recall

Of the articles from conspiracy websites, the share the model found. Like a metal detector: of all the metal buried, how much it found.

### Regularisation
Also: Penalty

A penalty that keeps weights small so the model cannot memorise too much. In scikit-learn, C is the inverse strength: small C means strong penalty. Like a teacher asking for short, simple answers. We choose C on validation.

### Sample
Also: Samples, Resample, Resamples

A smaller group picked from a bigger one. Our test articles are a sample of all articles, so scores on them are only an estimate. Like tasting one spoonful of soup to judge the whole pot.

### Sampling
Also: Temperature, default sampling

The randomness a language model uses when picking its next word. We keep the model's default settings, because it rejects a temperature setting. Like a die the model rolls, with a default weighting.

### Search words
Also: Topic filter, Search word

The fixed list of words and phrases (such as "9/11" or "World Trade Center") we use to decide whether an article is about the attacks. An article with none of them is removed as off-topic. Like keeping only the library books whose title mentions your subject.

### Shortcut learning
Also: Shortcut, Target leakage

A model scoring well by using a clue that is not what we care about, such as a website's name or layout. Like a student who passes a test by recognising the handwriting of the questions' author, not by knowing the subject. It is a form of leakage of the label into the input.

### Site concentration
Also: Concentration

How much of one class comes from a few websites. If three sites supply many conspiracy articles, their habits can sway what a model learns and how it scores. Like a survey where three loud people give most of the answers.

### Stance coding
Also: Stance, stance code

Reading an article and noting how it treats a conspiracy claim: endorses it, rejects it, reports it neutrally or makes no claim. A model did the coding here. Like a teacher marking each essay as for, against or neutral.

### Standardise
Also: Standardising, Standardised, Standardisation, Scaler

Rescaling a number feature so its average is 0 and typical spread is 1, using training articles only. Then no feature dominates just because its numbers are larger. Like converting heights and weights to the same scale.

### Statistical significance
Also: Significant

A difference is called significant when its p-value is below a limit we set beforehand, usually 0.05. It means unlikely to be chance, not important or large. Like a clear footprint: it shows someone was there, not who.

### Stratified

Splitting so that each part has about the same share of each class. If a quarter of all articles are conspiracy, about a quarter of every part is too. Like mixing a fruit basket so every bag gets the same share of apples.

### Style feature
Also: Style features

A clue about how an article is written rather than what it is about, such as sentence length, capital letters or exclamation marks. Like recognising a friend by their handwriting, not by what the note says.

### Test set
Also: Test split, Test articles

Articles locked away until the end and scored once for the final numbers. Like the real exam: it only counts if you have not seen the questions before.

### TF-IDF
Also: Term frequency, inverse document frequency, document frequency

A way to turn text into numbers that gives high scores to words frequent in one article but rare across all articles, and low scores to words like "the". Like a stamp collector who finds a rare stamp more telling than a common one.

### Threshold

A cut-off number that decides a yes or no. Our rule is a threshold: a group of clues counts as helping only if F1 rises by at least 0.01. Like a pass mark in an exam: 50 points pass, 49 do not.

### Token
Also: Tokens

A small piece of text a language model reads, often a word or part of one. Models limit how many tokens they may read and write. Like counting syllables instead of words.

### Training set
Also: Train split

The articles the model learns from. All learned quantities, such as the word list and the weights, come from these articles only. Like the practice questions a student studies before the exam.

### Two-sided test
Also: two-sided

A test that asks whether two things differ in either direction, not only in the direction we hoped for. Like checking whether a new recipe tastes different from the old one, better or worse, rather than only checking whether it is better.

### Unit length
Also: length normalisation, Unit-length

Scaling each article's list of numbers so that its overall size is exactly 1. Then a long article and a short one can be compared by their mix of words, not by how much they say. Like converting every recipe to "per 100 grams" so a big pot and a small pot compare fairly.

### Validation set
Also: Validation split

Articles kept aside to compare settings and choose the best one. The model never learns from them. Like a mock exam: you see how you do, then adjust your studying.

### Vector

A list of numbers that describes one article, one number per feature. Two articles with similar lists are similar. Like a row in a spreadsheet where each column is one clue.

### Vocabulary
Also: Word list

The list of words the model is allowed to use. We build it from training articles only, and drop very rare words. Like a dictionary a student makes from the course book alone, before taking the exam.

### Weight
Also: Coefficient, Weights

The number a model learns for each feature. A positive weight pushes towards conspiracy, a negative one towards mainstream. Like points on a scorecard. A high weight means useful for telling sites apart, not proof of anything.

### Word count
Also: Word counts

How many times each word appears in an article. "Inside" three times and "job" twice give two numbers. It is the simplest way to turn text into numbers, like a tally sheet for each article.

### Zero-shot

Asking a language model to do a task without showing it any worked examples first. Like asking a stranger to sort the letters after only one sentence of instructions.
