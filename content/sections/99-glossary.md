---
title: Glossary
part: Reference
hypothesis: Short plain definitions let every teammate follow the other sections without prior machine-learning knowledge.
---

Every technical word used on this site is explained here with an everyday example. The first use of each term in a section links to its entry.

### Ablation
Also: Ablation study

Taking one group of features out, or adding one in, to see what changes. Like removing one ingredient from a recipe to learn what it contributes.

### Accuracy

The share of all articles that the model labels correctly. It can mislead when one class is much bigger than the other. Like a student who guesses "true" every time on a quiz that is mostly true.

### Bag of words

Treating an article as a pile of words and ignoring their order. Like tipping a letter's words into a bag: you can count them but no longer see the sentences. Our first two analyses work this way.

### Baseline

A simple method we compare other methods against. If a fancy model cannot beat the baseline, the extra effort did not help. Like walking as the baseline against which a bicycle is judged.

### Bootstrap
Also: Bootstrap resampling, website bootstrap

Redrawing articles (or whole websites) at random with replacement, many times, and recomputing the score each time. The spread shows how shaky the score is. Like asking what a score would be on other, similar test sets.

### Cache
Also: Saved answers

A file that stores each answer the first time it is computed, so a rerun reads the file instead of paying again. Like writing a result in a notebook so you need not recalculate. It stores ids and codes, never article text.

### Certainty and uncertainty words
Also: Epistemic markers, epistemic words

Words that show how sure a writer is: "clearly" and "proves" for certainty, "maybe" and "allegedly" for doubt. We count them with a short hand-made list. Like noticing whether a friend says "I know" or "I guess".

### Class weight
Also: Balanced class weight

A setting that makes mistakes on the rarer class count more during training. "Balanced" weights each class by how rare it is. Like giving double points for finding the rarer kind of letter.

### Classifier

A program that puts each item into one of a few groups. Ours decides whether an article comes from a conspiracy-oriented or a mainstream website. Like a mail sorter who looks at each letter and drops it in one of two boxes.

### Cohen's kappa
Also: Kappa

A number showing how much two people agree beyond what chance would give. 1 is perfect agreement, 0 is no better than chance. Like two judges scoring the same dives.

### Confidence interval
Also: 95% confidence interval, interval

A range around a score that shows how uncertain it is. A 95% interval is built so that it would cover the true value in about 95 of 100 repeats of the study. Like a forecast saying "20 to 24 degrees".

### Confusion matrix

A small table that counts the right and wrong calls: conspiracy called conspiracy, conspiracy called mainstream, and the same for mainstream. Like a scoreboard showing which mistakes happen where.

### Corpus

A big collection of texts used for a study. Ours is the LOCO corpus, a set of web articles from conspiracy and mainstream websites. Think of a library shelf that holds only the books we study. The texts stay in a private data repository.

### Cosine similarity

A number from 0 to 1 that says how alike two articles are, based on their word vectors. Near 1 means almost the same words in the same proportions. Like comparing two shopping baskets by what they contain.

### F1 score
Also: F1

A single number between 0 and 1 that combines precision and recall, high only if both are high. We report it for the conspiracy class. Like a grade that averages two exams but punishes a very bad one.

### False positive
Also: False negative

A false positive is a mainstream article wrongly called conspiracy; a false negative is a conspiracy article wrongly called mainstream. Like a smoke alarm ringing without fire, or staying silent during one.

### Feature

One measurable clue about an article that a model can use, such as how often "hoax" appears or how long the sentences are. Like noticing a letter's stamp, colour and handwriting before guessing who sent it.

### Fine-tuning

Training an existing language model further on our own labelled articles. We did not do this. Like sending a trained assistant on a course just for our task.

### Frozen

Fixed once and never changed, such as the split of articles. We keep it frozen so every model faces the same exam. Like sealing the exam paper before anyone studies.

### Grid search
Also: Grid

Trying a short list of settings, such as several values of C, and keeping the one that scores best on validation. If the best is at the edge of the grid, an even more extreme value might do better. Like testing oven temperatures one by one.

### Grouped split
Also: Split by website

Dividing articles so that all articles of one website land in the same part (train, validation or test). Like putting whole families, not single people, into the same team, so a model cannot recognise a site it has already met.

### Hash
Also: Fingerprint

A short code computed from some input that changes completely if the input changes at all. We use it to tell whether a saved answer still matches the prompt. Like a fingerprint for a document.

### Hedges and boosters
Also: Hedge, booster

Hedges are words that soften a claim ("perhaps", "seems"); boosters make it stronger ("obviously", "undeniable"). Together they form our certainty and uncertainty word lists. Like saying "I think" versus "I am sure".

### Hypothesis

A clear guess about what we will find, written down before we look at the results. It can turn out wrong. Like predicting that a new recipe tastes better before anyone tastes it.

### Label
Also: Class

The correct answer we know for each article: conspiracy-oriented or mainstream website. It comes from the website, not from reading the article. Like the name of the sender written on every letter in a practice pile.

### Label noise

Labels that do not fit the article they are attached to. A mainstream site may publish a conspiracy-style piece, and the label still says mainstream. Like a shelf labelled "cookbooks" holding one novel.

### Leakage
Also: Data leakage

When information from the test articles sneaks into training, so the scores look better than they are. Like a student who has seen the exam questions in advance.

### Lexicon
Also: Hand-made word list

A hand-made list of words that belong to one theme, such as doubt words. We count how many of an article's words are on the list. Like a highlighter pen used only on one kind of word.

### LLM
Also: Large language model, language model

A program trained on huge amounts of text that writes answers to written questions. Ours is Claude Haiku 5.5. Like a very well-read assistant who has never met our corpus.

### Logistic regression

A simple model that gives each feature a weight, adds up the weighted values for an article and turns the total into a probability. Like a judge adding points for and against, then stating how likely the case is.

### Macro-F1
Also: Balanced accuracy

Scores that treat both classes as equally important, so the bigger class cannot hide poor results on the smaller one. Like grading each subject equally, not by how many questions it had.

### McNemar test

A test that compares two classifiers on the same articles by looking only at those where exactly one is right. It tells whether the difference is bigger than chance. Like two students answering the same exam, comparing their disagreements.

### N-gram
Also: Bigram, trigram

A run of n words in a row. Single words are unigrams, pairs such as "inside job" are bigrams, triples are trigrams. Like reading a phrase instead of one word at a time, which keeps a little of the word order.

### Near-duplicate
Also: Duplicate

An article that is almost the same as another one, such as the same story copied with small changes. We remove them so the model is not tested on what it has already seen. Like two photocopies with one line changed.

### Overfitting

When a model memorises quirks of its training articles and then does worse on new ones. Like a student who learns last year's answers by heart instead of understanding the topic.

### P-value

A number between 0 and 1 that says how surprising our result would be if there were really no difference. A small p-value (under 0.05) means chance alone is an unlikely explanation. Like how odd ten heads in a row would seem.

### Parse error
Also: Unreadable answer

A model reply we cannot turn into a clear answer, for example when it is cut off or off-format. Such articles need a rule. Like a form where the box for the answer is left blank.

### Precision

Of the articles the model called conspiracy, the share that really come from conspiracy websites. Like a metal detector: of all beeps, how many were really metal.

### Probability

A number from 0 to 1 saying how likely something is. A model output of 0.9 means it is fairly sure the article is from a conspiracy website. We call it conspiracy above 0.5. Like a weather forecast's chance of rain.

### Prompt
Also: Prompt variant

The written instruction we give a language model, including the question and the article. A different wording is a prompt variant. Like the exact question on a worksheet: small changes can change the answers.

### Random seed
Also: Seed

A starting number that makes a computer's random choices repeatable. With the same seed, the same split or the same shuffle happens again. Like fixing the order of a shuffled deck.

### Recall

Of the articles from conspiracy websites, the share the model found. Like a metal detector: of all the metal buried, how much it found.

### Regularisation
Also: C, penalty

A penalty that keeps weights small so the model cannot memorise too much. In scikit-learn, C is the inverse strength: small C means strong penalty. Like a teacher asking for short, simple answers. We choose C on validation.

### Sampling
Also: Temperature, default sampling

The randomness a language model uses when picking its next word. We keep the model's default settings, because it rejects a temperature setting. Like a die the model rolls, with a default weighting.

### Solver

The calculation method a program uses to find the best weights. Different solvers suit different data sizes. Like choosing between a hand saw and a power saw: both cut, one is faster for big jobs.

### Stance coding
Also: Stance, stance code

Reading an article and noting how it treats a conspiracy claim: endorses it, rejects it, reports it neutrally or makes no claim. A model did the coding here. Like a teacher marking each essay as for, against or neutral.

### Standardise
Also: Scaler

Rescaling a number feature so its average is 0 and typical spread is 1, using training articles only. Then no feature dominates just because its numbers are larger. Like converting heights and weights to the same scale.

### Statistical significance
Also: Significant

A difference is called significant when its p-value is below a limit we set beforehand, usually 0.05. It means unlikely to be chance, not important or large. Like a clear footprint: it shows someone was there, not who.

### Stratified

Splitting so that each part has about the same share of each class. If a quarter of all articles are conspiracy, about a quarter of every part is too. Like mixing a fruit basket so every bag gets the same share of apples.

### Style feature

A clue about how an article is written rather than what it is about, such as sentence length, capital letters or exclamation marks. Like recognising a friend by their handwriting, not by what the note says.

### Test set
Also: Test split

Articles locked away until the end and scored once for the final numbers. Like the real exam: it only counts if you have not seen the questions before.

### TF-IDF
Also: Term frequency, inverse document frequency, document frequency

A way to turn text into numbers that gives high scores to words frequent in one article but rare across all articles, and low scores to words like "the". Like a stamp collector who finds a rare stamp more telling than a common one.

### Token

A small piece of text a language model reads, often a word or part of one. Models limit how many tokens they may read and write. Like counting syllables instead of words.

### Tokenise
Also: Tokenisation

Splitting a text into small pieces, usually words, so they can be counted. "Inside job!" becomes two pieces: "inside" and "job". Like cutting a sentence into word cards.

### Training set
Also: Train split

The articles the model learns from. All learned quantities, such as the word list and the weights, come from these articles only. Like the practice questions a student studies before the exam.

### Type-token ratio

The number of different words divided by the number of all words in a text. A low value means the same words repeat a lot. A text saying "go, go, go" has few different words for its length.

### Validation set
Also: Validation split

Articles kept aside to compare settings and choose the best one. The model never learns from them. Like a mock exam: you see how you do, then adjust your studying.

### Vector

A list of numbers that describes one article, one number per feature. Two articles with similar lists are similar. Like a row in a spreadsheet where each column is one clue.

### Vocabulary
Also: Word list

The list of words the model is allowed to use. We build it from training articles only, and drop very rare words. Like a dictionary a student makes from the course book alone, before taking the exam.

### Weight
Also: Coefficient

The number a model learns for each feature. A positive weight pushes towards conspiracy, a negative one towards mainstream. Like points on a scorecard. A high weight means useful for telling sites apart, not proof of anything.

### Word count

How many times each word appears in an article. "Inside" three times and "job" twice give two numbers. It is the simplest way to turn text into numbers, like a tally sheet for each article.

### Zero-shot

Asking a language model to do a task without showing it any worked examples first. Like asking a stranger to sort the letters after only one sentence of instructions.
