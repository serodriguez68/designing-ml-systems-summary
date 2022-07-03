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

### Importance sampling

# Labeling

# Class Imbalance
