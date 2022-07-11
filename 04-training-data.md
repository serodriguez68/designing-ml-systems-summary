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
- Examples: 
	- Strong natural labels: 
		- ETA in google maps can be verified with the actual trip time. 
		- Stock price prediction can be verified. 
	- Weaker natural labels (usually need to be approximated):
		- Recommender systems typically allow for natural labels to be *approximated* by recording if a suggestion was taken (positive label) or if wasn't taken within a defined time window (implicit negative label).
		- In Newsfeed ranking problems, natural labels can be *approximated* by adding a like and dislike button. This is a case of explicit approximation.
- Companies tend to choose working on problems with natural labels while they are getting started with ML because they tend to be easier and cheaper problems to work on.

### Feedback loop length
In problems that have natural labels, the feedback loop is the time from when a prediction is offered to when we are able to derive or approximate the ground-truth label of that sample.
- If ground-truth labels is available within minutes, this is considered a short feedback loop. 
- Problems with shorter feedback loops are easier to work with and produce systems that adapt better to changing requirements and data distributions.
	- This is true for all problems. If you can get labels faster, your life will be easier.

#### Problems with strong labels
When dealing with problems that have strong natural labels, the length of this loop is usually determined by the nature of the problem. For example:
- Stock predictions can be verified within minutes.
- Google maps' ETA predictions can be verified in the order of hours.
- Fraud detection has a long natural feedback loop because the dispute process fully terminates months after a transaction is issued.

#### Problems where we approximate the labels
Trying to approximate a label typically brings some trade-off decisions to make.
- **Choosing the type of user feedback for approximation:** 
	- You can use different signals at different points of a user journey to approximate a label.  Different signals have different **volume, strength and loop length.**
	- For example in a product recommendation system you can use *"clicked on the product"* to generate a label. Alternatively you can use the *"bought the product"* signal. Clicks will happen more often and have a tighter feedback loop, but purchases are a much more valuable signal.
- **Choosing the time window for implicit approximation:**
	- Problems in which we need to implicitly infer a label because something *didn't* happen are very common. In these problems we are usually faced with the decision of choosing a **time window** after which the label is assigned. Recommender systems are a typical example.
	- Shorter time windows  = Shorter feedback loops + Higher mislabelling because the target action happened after the arbitrary time window limit.


## Handling the Lack of Labels


# Class Imbalance
