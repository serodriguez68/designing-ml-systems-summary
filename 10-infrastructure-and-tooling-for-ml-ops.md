# Chapter 10 - Infrastructure and Tooling for ML Ops

It is not uncommon to find data scientists that know the right things to do for their ML systems but that can't do them because their infrastructure does not support them. 
- When setup right, infrastructure can help automate processes, reducing the need for specialised knowledge and engineering time.
- When setup wrong, infrastructure is painful to us and expensive to replace.

ML infra can be grouped into 4 layers. This chapter will cover each layer. Note that chapter presents the layer in loose order of *"familiarity to an average developer"*  to ease comprehension.
![the four layers of ML infrastructure](10-infrastructure-and-tooling-for-ml-ops.assets/layers-of-ml-infrastructure.png)

The chapter ends with a discussion on how to navigate the **"build vs buy" decision**.


## Infrastructure requirements follow company scale
The infrastructure required for your team depends on how specialised your applications are:
- At one side of the spectrum there are companies that use ML for ad-hoc business analytics like quarterly reports or next year forecasts. 
	- The result of their work usually goes into a report or a slideshow. 
	- These companies don't need to invest in infrastructure as all they need is Jupyter Notebooks.
- At the other end of the spectrum, there are the companies that are pushing the envelope on the scale of use of ML. 
	- They have extremely low latency requirements, or they process petabytes of new data a day, or they need to do millions of predictions per hour. 
	- These are companies like Tesla, Google and Facebook. 
	- These companies usually have to develop their own specialised infrastructure. Some parts of this specialised infrastructure may be made publicly available later (like Google did through GCP).
- **The vast majority of companies live in the middle of the spectrum**. These are companies that use ML for *"common applications"*  at a *"reasonable scale"*.
	- *"Common applications"*:  fraud detection, churn prediction, recommender systems, etc.
	- *"Reasonable scale"*: 
		- Work with data in the order of gigabytes and terabytes, NOT petabytes.
		- The data science team has ten to a couple hundred engineers. NOT thousands.
	- These companies will likely benefit from using generalised ML infrastructure that has been increasingly standardised and commoditised. 
	- **This chapter focuses on the infrastructure needs of companies like this.**

![infrastructure requirements increase with company scale](10-infrastructure-and-tooling-for-ml-ops.assets/infra-requirements-vs-company-scale.png)



## Layer 1: Storage and Compute

## Layer 4: Development Environment

## Layer: 2 Resource Management

### Cron, Schedulers and Orchestrators

## Layer 3: ML Platform

### Model Store


### Feature Stores

## The Build vs Buy decision