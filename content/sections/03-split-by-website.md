---
title: Splitting by website
part: Evaluation
hypothesis: Keeping every website entirely inside one part gives a lower but honest score than a random split, and each part holds enough conspiracy websites to be usable.
---
## Explained simply

**Why we split the articles at all.** A student who sees the exam questions in advance will score well without having learned anything. A model is the same. If we test it on articles it has already learned from, the score proves nothing. So we divide the articles into three parts:

- **Training set (70%).** The model learns from these.
- **Validation set (15%).** We use these to compare settings and choose the best. Think of a practice exam.
- **Test set (15%).** We use these once, at the very end, to get an honest score. Think of the real exam.

**Why by website.** Articles from one website look alike: the same layout, the same typical words, the same topics. If some articles of a website are in training and others in the test set, the model can score well by simply recognising the website. Then we would not learn whether it understands anything about the texts. So we keep every website in one part only. A website in the test set is a complete stranger to the model, which is the situation we care about. Expect scores to be lower than with a random split, and more honest.

**Team question 1: does "grouped by website" mean the model is fed one website at a time?** No. The model sees all training articles mixed together. "Grouped" only decides which articles end up in the training set at all. The rule is: a website is never in two parts.

An example with three invented websites of ten articles each:

| website | label | random split | grouped split |
|---|---|---|---|
| site A | conspiracy | spread over all three parts | all ten in training |
| site B | mainstream | spread over all three parts | all ten in validation |
| site C | mainstream | spread over all three parts | all ten in test |

**Keeping the shares equal.** Conspiracy articles are fewer than mainstream ones. We divide the conspiracy websites and the mainstream websites separately, so each part has about the same mix. A fixed random number keeps the result repeatable. Because websites are not cut up, the parts are only close to 70/15/15.

**Frozen.** We made the split once, before any result existed, and saved it in a file. We never change it, because re-dividing after seeing scores would be a way of cheating. Cleaning later removes articles from the parts but never moves one to another part.

**What to look out for.** Only a few conspiracy websites are in the validation and test sets, so one unusual website can move a score noticeably.

## The code

This function gives each website to one part, largest websites first, so that the shares come out close to the targets.

<!-- cell: src/fml/split.py#assign_websites -->

This one builds the split file and refuses to overwrite an existing one.

<!-- cell: src/fml/split.py#make_split -->

<!-- output: results/tables/split_summary.csv | Websites and articles per part and class, after cleaning -->
<!-- output: results/figures/split_sizes.png | Articles per part and class, after cleaning -->
<!-- output: results/metrics.json#split.websites_in_more_than_one_split -->

## Report

The documents were divided into a training set, a validation set and a test set with target shares of 70%, 15% and 15%. The division was made by website: all documents of one website were assigned to the same set, so that no website appears in more than one set. This prevents a model from being rewarded for recognising the layout, vocabulary or topics of a website it has already seen, and it mimics the practical situation of classifying articles from unknown websites. The division was stratified by label: conspiracy and mainstream websites were assigned separately, from the largest to the smallest website, to the set that was furthest below its target share. A fixed random seed resolved ties. The assignment was made once, before any model was fitted, and stored in a file that was never changed; the cleaning steps only remove documents from it. The split file was made before cleaning, on all 3,146 documents; cleaning then removed documents from it without moving any to another set. The table shows the numbers of websites and documents that remain in each set after cleaning, and the second table the totals per set.

<!-- output: results/tables/split_summary.csv | Websites and documents per set and class (after cleaning) -->

Because websites cannot be divided, the shares only approximate the targets. Each set contains at least three conspiracy websites, and the number of websites in more than one set is zero. Validation and test sets still contain only a few conspiracy websites, so scores depend to some extent on which websites ended up there. All model settings were chosen on the validation set, and the test set was used only for the final comparison.

## Check yourself

Q: Why do we not test the model on the articles it learned from?
A: A model can remember its training articles, so the score would show memory and not the ability to handle new articles.

Q: What is the purpose of the validation set, and how does it differ from the test set?
A: The validation set is used again and again to choose the best settings. The test set is used once at the end, so it gives an honest final score.

Q: Why do all articles of one website go into the same part?
A: Otherwise the model could score well by recognising the website's typical style instead of learning something that carries over to new websites.

Q: Is the model trained one website after the other?
A: No. It sees all training articles together. Grouping only decides which articles belong to the training set.
