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

### The Storage Layer
Data storage has become so cheap that most companies just store all the data they have. We've discussed all the details about the storage layer in [Chapter 3 - Data Engineering Fundamentals](03-data-engineering-fundamentals.md) so we won't repeat them here.

The rest of this section will focus on the **compute layer.**

### The Compute Layer
- There are two common uses of the compute layer:
	- Execute jobs. In this case, compute units may only live for the length of the job.
	- Run Jupyter notebooks or other exploratory work. In this case, the compute units tend to be long-lived. They are usually called *virtual machines* or *instances*.
- Most commonly, the compute layer of a company is a managed cloud service like AWS Elastic Cloud or GCP.
- The most minimal compute layer would be a single CPU or single GPU that executes all your computation.
	- Some compute layer frameworks abstract away the notion of cores and use other units of computation. For example engines like Spark and Ray use "job" as their unit. Kubernetes uses "pod" as the smallest deployable unit.
- A compute unit is typically characterised by two metrics: **memory** and **operation speed**.
	- **Memory**: to execute a job, your unit first needs to load the required data into memory. The total memory determines how much data your unit can handle. The load and unload speed of data in and out of memory can also affect your total job time significantly. This is why cloud providers offer special instances that have **"high bandwidth memory"**.
	- **Operation speed:** the precise way to measure this is contentious and not uniform between cloud providers.
		- The most common metric is **FLOPS** - floating point operations per second. 
			- This is contentious because it is ambiguous what should be considered as a single "floating point operation". 
			- Also, it is nearly impossible to achieve 100% **utilisation rate** of your unit's rated FLOPs. 50% may be considered good.  Utilization rate depends load speed of data into memory speed.
		- [MLPerf](https://www.nvidia.com/en-us/data-center/resources/mlperf-benchmarks/#:~:text=MLPerf%E2%84%A2%20is%20a%20consortium,all%20conducted%20under%20prescribed%20conditions.) is a popular benchmark to measure hardware performance. It measures how long it takes the hardware to train common ML tasks.

#### Public Clouds VS Private Data Centers
- ML use cases tend to benefit from using public clouds **because ML workloads are bursty**. This means that you only pay for the compute during the work bursts and then free up the resources.
	- Cloud compute is elastic but not magical. It doesn't actually have infinite compute power and most cloud providers will impose limits on your account. You can often contact them and get those limits increased.
- **Early on**, using public clouds tends to give companies higher returns than buying their own storage and compute layers. However, **this becomes less defensible as the company grows**.
	- Cloud spending accounts for ~50% of the cost of revenue for large companies according to an a16z study.
	- The high cost of cloud has prompted large companies to start moving their workloads back to their own data centers.  This is called **cloud repatriation.**
		- Getting into the cloud is easy but moving away from it is very hard. Cloud repatriation requires non-trivial up-front investment in both hardware and engineering effort.

#### Multi-cloud strategies
- Multi-cloud means architecting your systems to use multiple cloud providers so that you can leverage the best and most cost-effective technologies of each provider and avoid vendor lock-in.
	- A common pattern in ML workloads is to do training on GCP or Azure, and then do deployment in AWS. Note that this is no necessarily a good pattern.
- Maintaining sanity in a multi-cloud strategy is very very hard. 
	- It is very hard to move data and orchestrate work-load across clouds.
- Often multi-cloud just happens by accident because different parts of the organization operate independently and make different choices. 
	- It is also not uncommon for ML companies to receive investments from parties that have interests in certain clouds and "force" the company to adopt that other cloud, resulting in multi-cloud arrangements.



## Layer 4: Development Environment

## Layer: 2 Resource Management

### Cron, Schedulers and Orchestrators

## Layer 3: ML Platform

### Model Store


### Feature Stores

## The Build vs Buy decision