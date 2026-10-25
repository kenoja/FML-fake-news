---
title: Splitting by website
part: Evaluation
hypothesis: We expect scores on whole unseen websites to be lower than on a random split but more honest, because the model never sees a website it is tested on.
---
## Explained simply

**Why we split the articles at all.** A student who sees the exam questions in advance will score well without having learned anything. A model is the same. If we test it on articles it has already learned from, the score proves nothing. So we divide the articles into three parts:

- **Training set (70%).** The model learns from these.
- **Validation set (15%).** We use these to compare settings and choose the best. Think of a practice exam.
- **Test set (15%).** We use these once, at the very end, to get an honest score. Think of the real exam.

**Why by website.** Articles from one website look alike: the same typical words and topics. If some articles of a website are in training and others in the test set, the model can score well by simply recognising the website. So we keep every website in one part only. A website in the test set is a complete stranger to the model, which is the situation we care about. Scores will be lower than with a random split, but more honest.

**Team question 1: does "grouped by website" mean the model is fed one website at a time?** No. The model sees all training articles mixed together. "Grouped" only decides which articles end up in the training set at all. The rule is: a website is never in two parts.

An example with three invented websites of ten articles each:

| website | label | random split | grouped split |
|---|---|---|---|
| site A | conspiracy | spread over all three parts | all ten in training |
| site B | mainstream | spread over all three parts | all ten in validation |
| site C | mainstream | spread over all three parts | all ten in test |

**Keeping the shares equal.** Conspiracy articles are fewer than mainstream ones. We divide the conspiracy websites and the mainstream websites separately, so each part has about the same mix. A fixed random number keeps the result repeatable. The parts are only close to 70/15/15.

**Why we also remove the website names.** The two steps do different jobs. The split stops the model from memorising a site. Removing names stops it from using a name in the text as an easy clue. Think of new exam questions, plus a covered pupil name for a fair marker.

**Frozen.** We made the split once, before any result existed, and saved it in a file. We never change it, because re-dividing after seeing scores would be cheating. Cleaning removes articles from the parts but never moves one to another part.

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

The documents were divided into a training set, a validation set and a test set with target shares of 70%, 15% and 15%. The division was made by website: all documents of one website were assigned to the same set, so that no website appears in more than one set. This stops a model from being rewarded for recognising a website it has seen, as when classifying unknown websites. The division was stratified by label: conspiracy and mainstream websites were assigned separately, from the largest to the smallest website, to the set that was furthest short of its target share, with a fixed random seed for ties. The assignment was made once and never changed.

<!-- output: results/tables/split_summary.csv | Websites and documents per set and class -->

Because websites cannot be divided, the shares only approximate the targets. The training, validation and test sets contain 25, 6 and 6 conspiracy websites (37 in total) and 54, 13 and 12 mainstream websites (79 in total); no website is in more than one set. All model settings were chosen on the validation set, and the test set was used only for the final comparison.

## Check yourself

Q: Why do we not test the model on the articles it learned from?
A: A model can remember its training articles, so the score would show memory and not the ability to handle new articles.

Q: What is the purpose of the validation set, and how does it differ from the test set?
A: The validation set is used again and again to choose the best settings. The test set is used once at the end, so it gives an honest final score.

Q: Why do all articles of one website go into the same part?
A: Otherwise the model could score well by recognising the website's typical style instead of learning something that carries over to new websites.

Q: Is the model trained one website after the other?
A: No. It sees all training articles together. Grouping only decides which articles belong to the training set.
