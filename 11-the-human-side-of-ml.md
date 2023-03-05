# Chapter 11 - The Human Side of Machine Learning

## User Experience in ML

We've discussed at length how software systems and ML systems are different. This means that there are additional UX challenges in ML systems that you need to consider.  This section overs 3 of those challenges.

### Challenge 1: Ensuring User Experience Consistency
- ML systems may give different predictions to the same user at different points in time. Changes in predictions can cause the UX of your system to feel broken to the user. 
- There are several reasons why a prediction might change from one time to the other.
	- ML systems are probabilistic by nature. This means that there is no guarantee that a given user will get the same prediction every time.
	- Your system may have different information about your user at different points in time, causing different predictions.
- When designing the UX for ML systems you will need to consider the **consistency-accuracy trade-off**:
	- You can decide to give the most accurate / updated prediction risking an inconsistent behaviour for the user OR you can decide to "remember and fix" what a prediction was using some rule to ensure consistency.

### Challenge 2: Combating "Mostly Correct" Predictions
- A "mostly correct" prediction is when an ML system produces predictions that are "mostly correct" or "look correct" but you  can't guarantee that they are. 
- Expert users in the domain will be able to discern and fix what parts of the prediction are wrong. Inexpert users may not have enough knowledge to discern and may incorrectly accept the given prediction as correct.
- This is very common with Large Language Models (LLMs). For example, you ask an LLM to produce some Java code to solve a problem and the LLM will output some code that looks like plausible Java.
	- An experienced programmer can quickly read the predicted code, fix a couple of things to make it work and use it.
	- An inexperienced programmer my attempt to use it "as is", even if the code is wrong.
- Systems that produce "mostly correct" predictions can be huge time savers for expert users. If your users are of this type, the UX concerns simplify.
- The harder UX challenge is how to correctly present "mostly correct" predictions to non-expert users so that the predictions are still useful but are also not taken at face value.
- A common approach is to show users multiple predictions from the same input to increase the chance of at least one of them being correct.
	- If you are dealing win non-expert users your predictions should be rendered in way that allows them to evaluate the variants. For example, if the output of the model are HTML alternatives, render each.
	- This approach is sometimes referred to "human -in-the-loop". Chip recommends reading ["Rethinking Human-AI interaction" By Jessy Lin](https://jessylin.com/2020/06/08/rethinking-human-ai-interaction/) for more info on this topic.

### Challenge 3: Smooth Failing
- No matter how tuned for speed your model is, you may run into the hard truth that, for certain conditions (like large inputs), your model may take an unacceptable amount of time to make a prediction.
	- This is typical of NLP or time series models: the bigger the text input, the longer it takes. 
	- High prediction latency can also happen for customers with a lot of data in your system. Don't disregard them as they may be your best customers.
- To address this you can:
	- Create a **backup model** that is less accurate but is faster. This backup model can be a 1) heuristic rule, 2)a simpler model or 3) a cached prediction with older data.
	- Decide on a strategy on how to **trigger the backup model.** This trigger can be: 
		- 1) add a timeout to the main model and fallback to the backup if the timeout is reached. 
		- 1) Add a heuristic rule that routes predictions. 
		- 3) Build an auxiliary regression model to predict how long the computation will take in the main model. If you do this, you need to take into account the inference of the regression prediction.
- Smooth failing  is related to the **speed-accuracy trade-off**. Sometimes you prefer to use a less-optimal model that is faster because latency is crucial.
	- With a primary + backup arrangement you can (sort of) choose both speed and accuracy.

## Team Structure
This section will cover aspects of team structure to consider when building ML teams.

### Don't disregard SMEs 
Subject matter experts (doctors, lawyers, farmers, stylists) are often overlooked in the design of ML systems or are only considered during the data labelling phase.

You should consider having SMEs as part of the team or at least involved in all phases beyond the labelling phase: development lifecycle (problem formulation, feature engineering, error analysis, model evaluation, user interface, etc). Their input can make or break your system.

It is important to allow SMEs to make contributions to the project without having to go through engineering for everything. For examples, many companies are building no-code/low-code platforms to allow SMEs to do changes in things like labelling, quality assurance and feedback. More platforms are being developed for other phases like dataset creations and issue investigation.

### Ownership boundaries for data scientists

Companies tend to follow one of these two approaches when deciding the "ownership boundaries" for data scientists.

#### Approach 1: Have a separate team to manage production
The data scientist / ML team develops the model in the [dev environment](10-infrastructure-and-tooling-for-ml-ops.md#Layer%204%20Development%20Environment). Then the model is given to a separate team (e.g. Ops/platform ML engineering team) to put it in prod and run it.

- **Pros**
	- It is easier to hire people with one set of skills than to hire people that know multiple skills at the same time.
	- Makes the lives of every individual easier as they only have to focus on their part of the chain.
- **Cons**
	- Communication and coordination overhead between teams.
	- Debugging becomes hard because when something fails it is not clear who to go to and people are not familiar with each other's code. You need to pull different teams to debug issues.
	- Fosters finger-pointing when things go wrong.
	- Everyone has a narrow context and no one has visibility into the entire end-to-end process. This means that no one sees high impact suggestions on how to improve it.

#### Approach 2: Data scientists own the entire end-to-end process
The data scientist / ML team has to worry about productionizing models.
- **Pros**
	- The opposite of the cons above.
- **Cons**
	- It is unreasonable to expect data scientists to know the low-level infrastructure skills required to put models in prod.
	- You may have a really hard time hiring people with all skills.
	- Your data scientist may spend more time writing infrastructure code than ML code.

#### Wait, if both approaches suck, what do we do?
Having ML teams or data scientists own the entire process end-to-end works better. However, to do this you need to provide them with **good high-level abstractions to deal with areas they are not experts on.**
- Abstract away the complexities like containerisation, distributed processing, automatic failover, etc.

In the "Netflix working model",  specialists from areas like infrastructure, build tools, deployment pipelines, metrics and dashboarding, (among others) first join the project to create tools that automate their parts.  Then the data scientists / ML team uses those tools to own the project end-to-end.
- In other words, build the tooling to develop the project end-to-end, then let the data scientists / ML team iterate using the tools.

![The Netflix working model](11-the-human-side-of-ml.assets/the-netflix-working-model.png)


## A framework for responsible AI

This section contains a framework your company can adopt to ensure that your models are responsible.

Keep into account that:
- This framework may not be sufficient for all use cases. Use your judgment.
- There are applications of AI that are unethical no matter the framework you use (e.g. criminal sentence decisions, predictive policing).

### Discover the sources for model biases
When developing a model, study the different sources of model biases. 

Biases can creep up in any stage of the model creation process. Here is a **non-exhaustive** list of places to check:
1. **Training data:** if your data is not representative of the real world, the presence or absence of data for certain groups can cause the model to bias against those groups. See [non-probability sampling](04-training-data.md#Non-probability%20Sampling).
2. **Labelling:** the more human annotators rely on subjectivity to annotate, the more biases you will get. Think about ways to ensure that annotators are following a standard guideline and  measuring the quality of the produced labels.
3. **Feature engineering:** Does your model use any features that can cause it to learn biases? Using features related to ethnicity, gender, race, sexual orientation, religion, etc is usually not recommended because of the high risk of biases. Using features that allow the model to infer these indirectly is also risky. See [invariance tests](06-model-development-and-offline-evaluation.md#Evaluating%20fairness%20with%20invariance%20tests) and [slice-based tests](06-model-development-and-offline-evaluation.md#Evaluating%20performance%20and%20fairness%20with%20slice-based%20evaluation%20tests).
4. **Model's objective:** Does your objective allow for fairness for all users or is it introducing a bias towards favouring majority groups or majority classes? 
	1. For handling majority classes see [Class Imbalance](04-training-data.md#Class%20Imbalance)
	2. If invariance tests and slice-based tests flag different results for different groups, you may want to reconsider your model framing or objective.
5. **Evaluation:** is your evaluation process introducing biases?
	1. If your evaluation is done using human evaluation, your model might be receiving bias through the humans (see labelling above).
	2. If your evaluation can be done automatically your model might get biased if you don't do [invariance tests](06-model-development-and-offline-evaluation.md#Evaluating%20fairness%20with%20invariance%20tests) and [slice-based tests](06-model-development-and-offline-evaluation.md#Evaluating%20performance%20and%20fairness%20with%20slice-based%20evaluation%20tests) or if you ignore certain groups during the tests.

### Understand the limitations of the data-driven approach
A data-driven approach to fairness and responsible AI is necessary **but it is not sufficient.**

Model developers need to understand at a human level how the lives of the users at the other side of the predictions will be impacted by them. By doing this you reduce the risk of having blind spots that come from relying too much on data.

### Understand the fairness trade-offs that happen when optimising your model for different properties

Often ML literature assumes that when you are making model changes to optimise for a property, all other properties remain static. This is not true. 

This assumption is particularly dangerous in the responsible AI space because many of the trade-offs **are yet to be discovered.**

Here we cover two example trade-offs that have been documented in literature: 
- Privacy vs accuracy trade-off
- Compression vs fairness-trade-off
If you are working with datasets that are **differentially private** or models that **are being compressed**, make sure you invest resources in studying these trade-offs to avoid unintended harm.

#### Privacy vs accuracy trade-off
- The higher the level of privacy the model provides, usually means the lower the model accuracy.
- The accuracy of differential privacy model drops much more for underrepresented classes and subgroups.

#### Compression vs accuracy fairness trade-off
- The idea of model compression is reducing model size at minimal accuracy cost. This topic is explained in [chapter 7](07-model-deployment-and-prediction-service.md#Faster%20Inference%20through%20Model%20Compression).
- It is not guaranteed that your compression will "spread out" the accuracy loss uniformly across all classes and subgroups. Compression can disproportionally affect underrepresented subgroups in the dataset.
- Not all compression techniques have the same level of disparate impact. Pruning incurs in far higher disparate impact than quantisation.
- Once again, [slice-based tests](06-model-development-and-offline-evaluation.md#Evaluating%20performance%20and%20fairness%20with%20slice-based%20evaluation%20tests) are a great tool to study the impact of compression.

### Act early

The earlier in the development cycle of an ML system that you can start thinking about how this system will affect the life of users and what biases your system might have, the cheaper it will be to address these biases.

### Create model cards
- Model cards are short documents that accompany trained models with info about how they were trained and evaluated. They also include context like intended use and known limitations. A model card template is shown below.
- The goal of model cards is to standardise ethical practice and allow stakeholders to reason about candidate models not only from the lens of performance, but also from the lens of responsible AI.
- Model cards are especially important in cases where the people who deploy and use the model are not the same as the people that developed it.
- Creating and updating model cards manually can be quite tedious.  It is important to invest in tools that can automatically generate as much of the model card as possible. Tensorflow, Metaflow and scikit-learn all have features for model cards. Some companies build their own model-card software.
	- Chip thinks model stores will soon evolve to support and generate model cards natively.

#### Example model card
- **Model details**: Basic info 
	- Person, team or organisation developing the model
	- Model date
	- Model version
	- Model type
	- Info about training algorithm, parameters, fairness constraints and features
	- Resources for more information (e.g papers, blog posts)
	- Citation details
	- License
	- Where to send questions or comments about the model
- **Indented use**
	- Primary intended users
	- Out-of-scope use cases
- **Factors**: factors could include demographic or phenotypic groups, environmental conditions, technical attributes or others
	- #todo: not clear to me what this is.
	- Relevant factors and evaluation factors
- **Metrics:** should be chosen to reflect potential real-world impacts of the model
	- Model performance measures
	- Decision thresholds
	- Variation approaches
- **Evaluation data:** details on the dataset(s) used for the quantitative analysis reported in the card.
	- Datasets
	- Motivation
	- Preprocessing
- **Training data:** If possible, mirror the evaluation data section above.
	- In practice, it is not always possible to include training data into the model card.
	- If including data is not possible, at least include distributions over various factors in the training datasets
- **Quantitative analyses**
	- Unitary results
	- Intersectional / slice-based results
- **Ethical considerations**
- **Caveats and recommendations**

### Establish company processes for mitigating biases
- Building responsible AI is a complex process. Ad-hoc processes leave a lot of room for error. It is important for companies to establish systematic processes for making their ML models responsible.
- Some companies do third-party audits
- Resources:
	- [Google's Responsible AI practices](https://ai.google/responsibilities/responsible-ai-practices/?category=fairness)
	- [AI Fairness 360](https://aif360.mybluemix.net/):  open source library with algorithms, metrics and explanations to help mitigate bias in models AND datasets.

### Stay up-to-date on responsible AI
Responsible AI is a fast moving field with new sources of bias constantly being discovered and new techniques developed.
Some resources to help you stay up to date:
- [ACM FAccT Conference](https://facctconference.org/index.html)
- [Partnership on AI]()
- [Alan Turing Institute's Fairness, Transparency and Privacy Group](https://www.turing.ac.uk/research/interest-groups/fairness-transparency-privacy)
- [AI Now Institute](https://ainowinstitute.org/)
