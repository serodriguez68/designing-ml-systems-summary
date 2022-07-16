# 4 - Training Data
This chapter covers different techniques for **creating good training data.**
- Sampling techniques to select data for training
- How to tackle common challenges with training data:
	- Labelling problems
	- Class imbalance problems
	- Lack of data problems and data augmentation

Creating training data is an iterative process. As your model evolves through a project lifecycle, your training data will likely also evolve. 

# Sampling
Sampling happens in different steps of the workflow. For example:
1. Sample from all possible real-world data to create good training data.
2. Sampling from a given dataset to create coherent train, test and validation splits.
3. Sampling from all possible events that flow though you ML system for monitoring purposes.

## Non-probability Sampling
Non-probability sampling is a bad idea. The samples selected are not representative of the real-world data and therefore are riddled with selection biases. This will result in your ML models performing poorly.

We describe some of the common non-probability sampling methods used so that you can identify them and avoid them.

### Convenience Sampling
Samples of data are selected based on their availability. This sampling method is popular because, well, it’s convenient. 

### Snowball Sampling
Future samples are selected based on existing samples. For example,  incrementally scrape twitter accounts by starting with a set of seed accounts and scraping who they follow. 

### Judgement Sampling
Experts decide what samples to include.

### Quota Sampling
Select samples based on quotas for certain slices of data without any randomisation. For example, force 100 responses from each of the age groups: under 30 years old, between 30 and 60 years old, and above 60 years old, regardless of the actual age distribution. 


## Probabilistic Sampling
Probabilistic sampling is a good idea. The specific type of sampling to use depends on your use case.

### Simple random sampling
Select `p%` of your population by giving each sample a `p%` chance of being selected.
- **Pros:** 
	- Very simple to implement.
	- Sampled data follows the distribution of the population.
- **Cons:** 
	- Rare classes have a high chance of being left out. Models trained on this sampled data may think that the rare classes do not exist.


### Stratified sampling
Divide the population into the groups you care about *and then* sample each group giving each entry a `p%` chance of being selected.
- **Pros:**
	- No matter how rare a label is, you ensure samples from it will be included in the selection.
- **Cons:**
	- Cannot be used if samples cannot be cleanly separated into groups. For example, in multi-label tasks. 


### Weighted sampling
Give each sample a different probability of getting selected. For example, all elements with label `A` have 50% chance, `B` have 30% chance and `C` have 20%.

In python you can use `weights` in `random.choice`
```python
import random 
random.choices(
	population=[1, 2, 3, 4, 100, 1000],
	weights=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
	k=2
) 
```

- **Pros:**
	- Can be used to incorporate **domain knowledge** into the ML system. For example, if you know more recent data is more valuable, give it more weight.
	- Can be used to **rectify incorrect distributions in the available data and make the sampled data match the real world distribution**. For example, let's say that your available data has 75% female and 25% male labels. However,  you know that the real wold distribution is 50/50%. To fix that you would use weighted sampling to rectify the distribution by giving more weight to male examples.
- **Cons:**
	- If not used carefully you can end up baking in biases into your data.
- **Note that:**
	- **Weighted sampling** is not the same as **sample weighting.** Sample weights are used during training to make certain samples affect the  algorithm's **loss function** more than others. Giving higher training weights to samples that are close to the decision boundary can help your algorithm learn the decision boundary better. 
	- #todo : how is weighted sampling related to [Class Imbalance](#Class%20Imbalance)?


### Reservoir sampling for sampling streams
Creating a continuous sampled dataset from stream of data has these challenges:
- You don't know the size of the population beforehand. However, following good practices, you probably want every sample in the dataset to have the same probability of being selected.
- You can't keep adding new samples to the sample dataset forever. At some point you will run out of memory.
- You must be able to stop the continuous sampling at any point in time and each element in the dataset must have been sampled with the correct probability.

The **reservoir sampling algorithm** allows you to continuously sample a dataset of `k` elements while overcoming the challenges above. `k` is limited by the amount of memory you have.

1. Put the first `k` elements into the reservoir. 
2. For each incoming `nth` element, generate a random number `i` such that `1 ≤ i ≤ n`. 
3. If `1 ≤ i ≤ k`: replace the `ith` element in the reservoir with the `nth` element. Else, do nothing. 


# Labelling
Good labelling is a critical part. Some companies even have in-house labelling teams (e.g. Tesla). 

Hand labelling is full of problems, many of which have no good solution. If your problem allows, circumvent these problems by either using [Natural labels](#Natural%20labels) or the techniques described in [Handling the Lack of Labels](#Handling%20the%20Lack%20of%20Labels) .

## Hand labels
- **Problem 1: Hand labelling is expensive**. Especially when you need subject matter experts to produce the labels (like medical Doctors classifying X-rays).
- **Problem 2: Hand labelling poses a threat to data privacy.** Someone will need to have access to non aggregated  data. This makes it harder to outsource labelling and adds a security assurance burden.
- **Problem 3: Hand labelling is slow.** Slow labelling leads to slow iteration speeds, which in turn makes your models less adaptive to changing environments. The longer the labelling process takes, the more you existing model performance will degrade.
- **Problem 4: Label multiplicity**: Having different annotators label data from different data sources can lead to label ambiguity. 
	- Disagreements between annotators are extremely common.
	- The higher the level of domain expertise required, the higher de potential for disagreements.
	- **Partial Solution:** The best way to minimise this is to put great care in providing annotators with a clear problem definition and instructions.
- **Problem 5: Data lineage tends to be forgotten:** For example, you train a good model using 100K samples with good labelling quality.  Then you outsource another 1M samples for labelling. Unfortunately, the quality of the outsourced labels is poor (but you don't know it). Then you mix the data and train the same model with 1.1M samples. Finally you discover that your model performance has decreased as a consequence of the poor labelling. To make things worse, you cannot recover from it because you have mixed the data.
	- **Solution:** Use *data lineage* practices. Keep track of the origin of each of your data samples as well as the origin of their labels. Tracking this can help you debug data problems.

## Natural labels
Some problems have **natural ground  truth labels** that can be automatically derived **or approximated** by the system.  
- Some problems have stronger natural labels than others. 
- Label approximation can either be done using **explicit** or **implicit**  labelling. Explicit labelling means that we ask the user to give us the label in some way.
- Examples of strong natural labels: 
	- ETA in google maps can be verified with the actual trip time. 
	- Stock price prediction can be verified. 
- Examples of weaker natural labels (usually need to be approximated):
	- Recommender systems typically allow for natural labels to be *approximated* by recording if a suggestion was taken (positive label) or if wasn't taken within a defined time window (implicit negative label).
	- In Newsfeed ranking problems, natural labels can be *approximated* by adding a like and dislike button. This is a case of explicit approximation.
- Companies tend to choose working on problems with natural labels while they are getting started with ML because they tend to be easier and cheaper problems to work on.

### Feedback loop length
In problems that have natural labels, the feedback loop is the time from when a prediction is offered to when we are able to derive or approximate the ground-truth label of that sample.
- If ground-truth labels is available within minutes, this is considered a short feedback loop. 
- Problems with shorter feedback loops are easier to work with and produce systems that adapt better to changing requirements and data distributions.
	- This is true for all problems. If you can get labels faster, your life will be easier.

#### Problems that have strong labels
When dealing with problems that have strong natural labels, the length of this loop is usually determined by the nature of the problem. For example:
- Stock predictions can be verified within minutes.
- Google maps' ETA predictions can be verified in the order of hours.
- Fraud detection has a long natural feedback loop because the dispute process fully terminates months after a transaction is issued.

#### Problems that need label approximation
Trying to approximate a label typically brings some trade-off decisions to make.

##### Choosing the type of user feedback for approximation
You can use different signals at different points of a user journey to approximate a label.  Different signals have different **volume, strength and loop length.**

For example in a product recommendation system you can use *"clicked on the product"* to generate a label. Alternatively you can use the *"bought the product"* signal. Clicks will happen more often (i.e. more data) and have a tighter feedback loop, but purchases are a much more valuable signal.

There is no right decision here. You need to weight the tradeoffs and make a call.

##### Choosing the time window for implicit approximation
Problems in which we need to implicitly infer a label because something *didn't happen* are very common. In these  we usually need to choose a **time window** after which the negative label is assigned (e.g. User didn't watch the recommended movie).
- Shorter time windows  = Shorter feedback loops + Higher mislabelling because the target action happened after the arbitrary time window limit.


## Handling the Lack of Labels
This section covers 4 categories of techniques that have been developed to deal with the challenges of getting sufficient high-quality labels. You may use one or several of them at the same time.

Overview:
![](04-training-data.assets/handling-lack-of-labels-overview.png)

### Weak Supervision Labelling

This youtube video explains several arrangements for weak supervision, including some iterative arrangements that are very interesting.
https://www.youtube.com/watch?v=SS9fvo8mR9Y


**The TL/DR of it is:**

The idea of weak supervision is motivated by the assumption that systems trained using a lot of data tend to perform better than models with a small but perfect dataset; even if the labels you use for training are somewhat noisy. This is especially true if your model is a re-train or a fine-tune of some pre-trained model, like a previous iteration of the same model or doing transfer learning model (like a language model). 

The high level steps are:
1.You (or a subject matter expert) develop a set of heuristics to label data automatically using some simplifications. You wrap each of those heuristics inside a `labelling_function` (LF). Different heuristics might contradict each other and some might be better at labelling certain samples than others. All of this is ok.
```python
def labelling_function(example_note):
  if "pneumonia" in example_note:
	return true
```
- Some types labelling functions:
	- Keyword heuristics: is a keyword in the sample?
	- Regular expressions: does the sample match a regex?
	- Database lookup: does the string contain a string that matches the list of dangerous diseases?
	- Outputs from previously trained models. They can either be simple models trained on some small set of hand-labelled data or the output of a previous iteration of a model.
2. You apply the set of labelling functions to the data you want to label.
3. A tool like [Snorkel](https://github.com/snorkel-team/snorkel) will be able to take in all the different "label votes" produced by the labelling functions, learn the correlations between the them and output a probability vector of the label (e.g. [80% black, 10% green, 10%white]). Essential what Snorkel is doing is combining, de-noising and re-weightings the votes from  all LFs to obtain **the label likelihoods.** 
4. You would then train your big model using the output probability vectors.

- **Pros:**
	- **No hand labels required:** In theory you don't need hand labels for weak supervision. However having a small sample of hand labels if recommended to check the accuracy of your LFs.
	- **Privacy:** Weak supervision is very useful when your data has strict privacy requirements. You only need to see a few samples to create an LF, the rest can be done automatically.
	- **Speed:** Labelling lots of data with LFs is very fast (when compared to hand labelling)
	- **Adaptability:** when changes happen, you can just change your LF and reapply it on all of your data.  If you were hand labelling, you would need a full re-label. 
	- LFs allow you to incorporate subject matter expertise and to **version it, reuse it and share it.**
- **Cons:**
	**- Too Noisy labels:** Sometimes labels can be too noisy to be useful. To make things worse, unless you have some small set of hand labelled data, you won't know how bad the noisy labels are.


### Semi-Supervision
Conceptually, semi-supervision is using an initial set of labels as a seed for labelling more unlabelled data using some sort of method.

Here are 3 examples of semi-supervision implementations:

1. **Self-training**: use the seed of labelled data to train a model > use that model to produce labels for some unlabelled data > add samples with high raw probability to the training set > rinse and repeat until you are happy with your model's performance.
3. **Labelling by clustering:** assume that the unlabelled data points that cluster close to labelled data points share the same label.
4. **Perturbation:** It assumes that small perturbations to a sample shouldn't change its label. So you apply small perturbations to your training instances to generate new training instances. This can also be considered as a form of [Data Augmentation](#Data%20Augmentation).

### Transfer learning
A model developed for a task is reused as a starting point for a model a model on a second task. The first task usually has cheap and abundant training data.  
- Transfer learning is based on the idea that large neural models tend to be very robust against changes in task. For example, a language model for the English language trained on Wikipedia will be useful even if your NLP task has nothing to do with Wikipedia. 
- Language models built on large corpuses are a typical examples of transfer learning (e.g. BERT).
- Fine tuning the base model for your task can mean making small changes to the base model, such as continuing to train all or part of the model with your own data for your own task.
- Transfer learning has gotten a lot of interest because it allows you to use for free models that could have cost tens of millions of dollars to train.
- Usually the larger the pre-trained base model, the better its performance on downstream tasks.

### Active learning
Active learning is based on the idea that there are some samples that are more valuable to label than others. For example, samples that are closer to the decision boundary are more valuable because they will allow your model to learn the boundary better.

In active learning, your model will automatically tell you which samples you should go and label. Some examples of active learning implementations are:
1. You apply your model to a set of unlabelled data and your model selects the samples with less certainty (e.g less raw probability) to be labelled and added to the training data.
2. You apply ensemble of models to a set of unlabelled data and select the samples with less consensus for labelling.  The ensemble can be made of models trained on different slices of data, or with different hyperparams, or different models altogether. 
3. Some active learning techniques allow your models to synthesise samples that are in the region of uncertainty. 
4. Some companies apply active learning with the data being evaluated in production. If a model running in prod is not very certain about a sample it just saw, the sample is flagged for labelling.

# Class Imbalance

# Data Augmentation
