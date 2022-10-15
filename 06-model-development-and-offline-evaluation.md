# 6 - Model Development and Offline Evaluation


With an **initial** set of training features, model development can finally begin.  Keep in mind that feature engineering and model development are an iterative process.

This chapter will cover:
- Model selection, development and training
	- Aspects to consider when making a model selection decision
	- Ensembles as part of model selection
	- Experiment tracking and versioning during model development.
	- Distributed training
	- AutoML
- Offline evaluation: How to evaluate your model alternatives to pick the best one
	- Baselines
	- Sanity checks to try before sending your model to production

## Model Selection, Development and Training
### Criteria to select ML models
There are many aspects you need to balance out when making a model selection decision. Model selection is an art more than a science. This chapter should give you some guidelines and tips on what to consider.

#### Tip 1: The right tool for the right job
Narrow the search space of model selection by focusing only on the models that are suitable for the problem at hand. For example:
- If you want to build a fraud detection system, you should know this is a classic anomaly detection problem and that common models used for this category of problems are KNNs, isolation forest, clustering and neural networks.
- If you are tasked with an NLP text classification problem your options typically are naive-Bayes, logistic regressions, recurrent neural networks and transformer-based models such as BERT or GPT.

#### Tip 2: Consider model properties beyond training metrics
Direct training-related performance metrics like accuracy, F1, log loss are typically used for model selection. However, these should not be only metrics to consider.  

You should also care about other properties like amount of data needed, computational needs of the model, time to train, inference latency and model interpretability.  

The set of metrics you select to make your decision varies depending on your problem. For example, some in some use cases latency is critical even if it means sacrificing a bit of accuracy.

#### Tip 3: Avoid the state-of-the-art trap
Models with "state-of-the-art"  performance  usually come directly from research. However researchers often only evaluate their models in academic settings using existing pre-defined datasets.

The fact that a model has state-of-the-art performance in research does not mean that the model will be fast enough, or cheap enough *you* to implement. Similarly it doesn't mean it will has the same performance on *your data*.

If there’s a solution that can solve your problem that is much cheaper and simpler than state-of-the-art models, use the simpler solution.

#### Tip 4: Avoid the "classic ML is dead" trap
Neural networks receive a lot of press these days. However, that doesn't mean that classical ML models are going away. Classical models are still vastly used in production, especially when latency or explainability are important.  

Additionally, it is not rare to see classical and neural models deployed in tandem.  For example:
- Ensemble arrangements using neural models and decision trees.
- A classical model like k-means can be used to extract features that become input to a NN.
- Pre-trained neural networks like BERT of GPT-3 are used to generate embeddings that are fed to classical models like a logistic regression.

Don't discard classical models just because the are not "neural networks".

#### Tip 5: Start with the simples model
- Simpler models are easier to deploy. Deploying your model early allows you to validate things like: 
	- End-to-end behaviour of your feature
	- Consistency between your prediction and training pipelines
	- Your system's ability to detect [natural labels](04-training-data.md#natural-labels) (if your problem has any)
	- Your system's ability to detect data distribution shifts and re-training triggers
- Starting simple and adding more complex components slowly makes it easier to understand and debug models.
- Simple models become a baseline to compare complex models against.
- Simple models and low-effort models are different things.  Using a pre-trained BERT model is low-effort (because the work is done for you), but the model is not simple. 
	- Some times it makes sense to start with low-effort  & complex models instead of simple models. However, even in those cases having a simple model as a baseline is valuable because improving upon a complex model is very challenging. 

#### Tip 6: avoid human biases in selecting models
%%YOU ARE HERE%%