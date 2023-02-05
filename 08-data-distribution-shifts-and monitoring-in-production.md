# 8 - Data distribution shifts and monitoring in production

Deploying the model to production is NOT the end of the story. Models in production need to be monitored continually to detect the natural performance degradation that all models go through over time.

This chapter covers:
- The two different ways ML models fails: software system failures VS ML-specific failures.
- The chapter then does a deep dive into a particularly thorny ML-specific failure: **data distribution shifts.** It covers the type of data distribution shifts and how to detect them and address them.
- The chapter finishes with **monitoring and observability**:
	- Operational metrics that any software system should have (e.g. latency, CPU utilization)
	- ML-specific metrics

The [next chapter](09-continual-learning-and-test-in-production.md) covers how to fix your model when issues are detected.

# Causes of ML System Failures
ML systems can fail in two broad ways: as a software system and ML-specific failures.

## Software system failures
ML systems are also software distributed systems so all software failure modes apply. Some common ones are:
- **Dependency failure:** Something you depend on breaks and your program breaks as a consequence.
- **Deployment failure:** for example deploying the wrong version of the ML artefact or deploying a bug in the code surrounding the ML model.
- **Hardware failure:** The infra you run your model in fails and your system fails with it.

Google did an internal survey on the last 15 years of ML system failures and found that 60 out of 96 were software failures and not ML-specific failures. Furthermore, the found that most of the 60 were failures related to distributed system failure modes. 

As of 2022 Software system failures in ML systems are still very prevalent because the practices, standards and tooling for managing these systems is still nascent compared to traditional software. We can expect this to diminish over time as the field matures.

Additional Resources:
- To know more about this topic Chip recommends ["Reliable Machine Learning" by Todd Underwood](https://oreil.ly/5UOds).

## ML-specific failures

Examples are:
- Data collection and processing problems
- Poor hyperparameters
- [Changes in the training / inference pipeline not replicated on the other one]([07-model-deployment-and-prediction-service](07-model-deployment-and-prediction-service.md#Streaming%20Prediction%20Unifying%20the%20Batch%20Training%20Pipeline%20with%20the%20Streaming%20Serving%20Pipeline))
- Data distribution shifts (more on this below)
- Edge cases
- Degenerate feedback loops

This section will cover two of them and the next section will focus exclusively on data distribution shifts.

### Extreme data sample edge cases
Edge cases are data samples so extreme that cause the model to make catastrophic mistakes.

In some applications, edge cases can prevent your model from being deployed altogether if the consequences of a prediction gone wrong are catastrophic. A typical example is self driving cars.

Edge cases and outliers are related yet different things. An outlier is a data sample that is extreme, but that the model is able to handle.

There is a trade-off during training with excluding outliers and model robustness against edge cases: outliers are usually removed during training to help the model learn the decision boundaries better. However, removing outliers can also lead to surprising outcomes when extreme data samples are seen at inference time (where outliers cannot be removed).

### Degenerate feedback loops

A degenerate feedback loop can happen when the predictions themselves (e.g. the recommended song) influence the feedback (e.g. the chosen song), which, in turn, influences the next iteration of the model.

For example, imagine a resume screening model that is running in production that learnt that the feature "worked at Acme Inc" is a good predictor of performance.  The model disproportionally surfaces ex-Acme Inc resumes to recruiters and therefore more ex-Acme Inc people get hired than any other company. This will cause a future iteration of the model to put even more weight to the "worked at Acme Inc" feature. If left unchecked, the model will bias itself and real business performance will suffer.

Degenerate feedback loops only happen when a model is in production and users are interacting with it. Degenerate feedback loops are especially common in tasks with natural labels from users, such as recommender systems and ads click-through-rate prediction. 

This is a heavily researched area. You can find it under names like “exposure bias,” “popularity bias,” “filter bubbles,” and sometimes “echo chambers.”

#### Detecting degenerate feedback loops
To detect if your model is suffering from a degenerate feedback loop you measure the model output's **diversity**. There are several diversity-related metrics:
- Aggregate diversity and average coverage of long-tail items
	- Brynjolfsson et al. (2011), Fleder and Hosanagar (2009), and Abdollahpouri et al. (2019)
- Hit rate against popularity:  Measure the prediction accuracy of your system for different popularity buckets. If a recommender system is much better at recommending popular items than recommending less popular items, it likely suffers from popularity bias.
	- Chiea et al. (2021)
- Doing [feature importance studies (as described in chapter 5)](05-feature-engineering.md#Feature%20Importance) can help you detect over time if a model is biasing itself by giving more and more weight to a feature.  

#### Correcting degenerate feedback loops

##### Method 1: use randomization
Introduce randomization to model outputs to reduce homogeneity. For example, instead of showing the user only the items that the model recommends, also show the user random items and use their feedback to determine the true quality of the items. 

TikTok follows this approach. Each new video is randomly assigned an initial pool of traffic. This traffic is used to evaluate the video's unbiased quality to determine whether it should receive a bigger pool of traffic or be marked as irrelevant. 

**Randomization improves diversity at the expense of user experience.**  A more sophisticated strategy is to use  [contextual bandits](09-continual-learning-and-test-in-production.md#Contextual%20bandits%20as%20an%20exploration%20strategy) to determine when to explore vs exploit and increase diversity with a predictable accuracy loss. 

##### Method 2: use positional features
If the position of a prediction influences how likely it is to be clicked on, then you can try to teach the model the influence of position using *positional features*. 
\*Note that *positional features* are different from *"positional embeddings"* as described in [chapter 5](05-feature-engineering.md#Discrete%20and%20continuous%20positional%20embeddings).

Positional features can be numerical (1, 2, 3, ...) or boolean (e.g. was this item the first prediction?).

A naïve example to illustrate the point:
1. Add a boolean positional feature to your training data as shown below
2. During inference, you want to predict if a user will click on a song regardless of the song being recommended in 1st position or not. To do that you set the `1st-position` feature to false for all your candidates

![using positional features to mitigate degenerate feedback loops](08-data-distribution-shifts-and%20monitoring-in-production.assets/positional-features-and-degenerate-feedback-loops.png)

Note thats this naïve example will probably not be good enough to combat a degenerate feedback loop. Fancier multi-model systems can be used but the underlying idea is the same. 

##### Method 3: Use contextual bandits
Read more about bandits and contextual bandits in [Chapter 9](09-continual-learning-and-test-in-production.md#Side%20note%20Using%20contextual%20bandits%20to%20improve%20recommendation%20algorithms)

# Data Distribution Shifts

This is a type of [ML-specific failure](08-data-distribution-shifts-and%20monitoring-in-production.md#ML-specific%20failures) that deserves its own section because of how hard it is to detect and act on.

Some terminology:
- *Source distribution:* distribution of the training data
- *Target distribution:* distribution of the inference data in production

Data distribution shifts refer to differences between the *source distribution* and the *target distribution*.

**IMPORTANT: Data distribution shifts are only a problem if they cause your model's performance to degrade.** The fact that you have one does not necessarily mean you need to act on it.

## Types of Data Distribution Shifts

In theory, not all data distribution shifts are equal. However, in practice:
- Determining which type of shift occurred can be very hard. 
- The way industry engineers deal with distribution shifts  tends to be the same regardless of the type of underlying shift. 

A model might suffer from multiple types of drift at the same time.

### Covariate Shift
P(X) is different between the source and target distributions but P(Y|X) remains the same. In other words, the distribution of the inputs change but the probability of a label given a specific input remains the same.

Example: You build a breast cancer detection model using data from a cancer hospital. The hospital data contains more data of women over 40 years of age than seen in the general population because these women are encouraged to get this exam as a routine exam. That is,  P(X=women over 40) is different in source and target distributions.  However, the P(Y=breast cancer|X=woman over 40) is the same regardless of whether the women is part of the training data or not. 

Covariate shifts can happen for several reasons:
- At training time (affecting the *source distribution*):
	- Sample selection bias: like the example above.
	- Training data is  artificially altered to assist in model training (like discussed in [Chapter 4 > Data level methods for handling class imbalance](04-training-data.md#Data-level%20methods%20resampling)).
	- The learning process is altered through  [active learning](04-training-data.md#Active%20learning) to expose the model more to samples that are hard to classify. This changes the underlying *source distribution* the model learns from.
- In production (affecting the *target distribution*):
	- Usually a result of a major change in the environment your application is used. For example: a new demographic of users is added, you launch in a new country.

### Label Shift
Aka: prior shift, prior probability shift, target shift.
P(Y) is different between the source and target distributions but P(X|Y) remains the same.  Following the example above, P(Y=has breast cancer) is different in the source and target distributions but P(X=is over 40|Y=has breast cancer) is the same in both distributions.

Covariate shifts can also cause label shifts (like with the example above) but not all label shifts are caused by covariate shifts.  See the book for an example on the latter.

### Concept Drift
aka: posterior shift, "same input, different output"

P(Y|X) changes but P(X) remains the same. For example, you trained an apartment price estimator using data pre-Covid. If you use that same model post Covid, it would suffer from a serious concept drift, because the apartments are the same (i.e P(X=apartment features) is the same pre and post covid), but the price they are valued at has changed dramatically. 

In may cases concept drifts are cyclic or seasonal. For example, ride share prices fluctuate on weekdays vs weekends.  Companies might have different models trained on different seasonality data to reduce concept drift (e.g. model for weekends vs model for weekdays).

### Label Schema Change

In classification tasks this happens when your model was trained to output `N` amount of classes and your business requirements change and now you need to predict `N+m` classes. This is often the case with high-cardinality classification problems.

In regression tasks this happens when the range of the output variable changes.


## Detecting Data Distribution Shifts

**Data distribution shifts are only a problem if they cause your model's performance to degrade.**

### Detection using accuracy-related metrics
The best mechanism for detecting data distribution shifts is to monitor your model's [accuracy-related metrics](08-data-distribution-shifts-and%20monitoring-in-production.md#Monitoring%20accuracy-related%20metrics) (e.g. accuracy, F1, recall, AUC-ROC, etc) in production. 
- If there is a big difference between accuracy calculated using the test set at training time and the observed production accuracy, then you may have a data shift problem. 
- If the observed production accuracy changes over time, then a data drift problem may have appeared that was not there in the past (e.g. a seasonal concept drift, change of user base).

If you have access to [natural labels](04-training-data.md#Natural%20labels), then detecting shift using accuracy-related metrics is ideal. Unfortunately, in production you don't always have access to labels. Even if you do,  labels may be delayed beyond a reasonable time window to make accuracy monitoring useful. If that is the case, you still can detect data distribution shifts using statistical methods.

### Detection using statistical methods

Strictly speaking you should be interested in monitoring the the input distribution P(X), the actual label distribution P(Y), and the conditional distributions P(X|Y) and P(Y|X).

The hard thing, is that you need ground truth production labels to be able to monitor P(Y) P(X|Y) and P(Y|X). Furthermore, if you had access to ground truth labels, you would probably be better off doing detection by using accuracy-related metrics.

This is why in practice, most of the industry focuses on monitoring and detecting shifts in the [distribution of predictions P(Y_hat)](08-data-distribution-shifts-and%20monitoring-in-production.md#Monitoring%20predictions) and the [ input distribution P(X)](08-data-distribution-shifts-and%20monitoring-in-production.md#Monitoring%20features) if they don't have access not ground truth labels.

#### Drift detection through simple descriptive statistics

This is a simple, good start.

To figure out if the *source* and *target* distributions have shifted, calculate descriptive statistics (like min, max, median, variance, various quantiles, skewness, kurtosis, etc) for the training set and for the seen production data. 

If the descriptive statistics are very different, chances are there has been a distribution shift. However, the opposite is not true: **having similar statistics does NOT guarantee that there has been no shift.**

#### Drift detection through hypothesis tests

A more sophisticates solution is to use statistical tests designed to test whether the difference between two populates is statistically significant. 

Some pointers when doing this:
- Having statistical difference in the tests does not mean that in practice the difference is important. Again, shifts become problematic when they hurt your performance.
- If you are able to detect the statistical difference through a test using a small sample, then it probably means that the difference is serious. On the other hand, if it take a huge amount of data to detect the statistical difference, then the difference is possibly very small and not worth worrying about.  
- Two-sample tests often work better on low-dimensional data. It is highly recommended that you reduce the dimensionality of your data before running a test on it.
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) is a great open source package with python implementations of many drift detection algorithms.

Some tests that can be used for this (table taken from Alibi Detect):
![Alibi detect drift detection algorithms](08-data-distribution-shifts-and%20monitoring-in-production.assets/alibi-detect-drift-detection-algorithms.png)
- Kolmogorov-Smirnov test: 
	- It is good because it does not require any parameters of the underlying distributions to work and does not make any assumptions about the underlying distributions (so it works for any distribution).
	- It won't work for high-dimensional data, which unfortunately is often the case.

### Time window considerations for detecting shifts

When comparing your training data distribution (aka source distribution) against your production data distribution (aka target distribution), you will need to make a choice of the **time window** you will use to get production data to run your tests.  This section contains some things to consider when choosing that **time window**.
- Consider the seasonality of your data. If your data has a natural weekly cycle, and your training data contains multiple weeks, choosing a production **time window** that is less than a week could result in weird results.
- Consider the speed of detection vs reliability of the test trade-off. Shorter time windows can help you detect shifts faster. However, they can also result in more false alarms. Longer time windows have the opposite characteristics.
- Keep in mind the differences between *accumulating time windows* and *sliding time windows.* *Accumulating time windows* keep adding data to the production data set to test as time progresses without "discarding" any of the old data on the other end. *Sliding time windows* discard data that has fallen outside the time window.
	- *Accumulating time windows* can have significantly more data and therefore the tests can be more reliable. However, this also means that tests run on them are less reactive to sudden changes because the data that is already there may obscure recent changes.
	- *Sliding time windows* have the opposite characteristics.





## Addressing Data Distribution Shifts

### Minimizing model sensitivity to shifts
So far, we have discussed that data distribution shifts are inevitable. However, it is also true that there are certain things you can do make your model less sensitive to shifts.

#### Technique 1: Train your model using a massive dataset
The hope here is that by using a massive dataset, the model will learn such a comprehensive distribution that whatever data points the model encounters in production will likely come from this distribution.

This is more common in research and is not always possible in industry. Nevertheless, it is worth mentioning it.

#### Technique 2: Consider the trade-off between performance and stability when selecting features
Some features are more prone to distribution change than others. For example, app store ratings get reset with every app and OS version release. You may want to consider using coarser percentile ratings instead. The feature may not be as nuanced but it will be more stable and robust against changes.

#### Technique 3: Consider creating separate models for fast-moving markets and slow-moving markets
Imagine, you are tasked with building a regression model for house prices in the US. By exploring your data, you notice that price changes in San Francisco and New York change much more rapidly than the rest of the country. By creating independent models for those cities, separate from your main model, you reduce the need for constantly retraining your main model. Additionally, your fast market models can keep up to date with more frequent re-trainings.

### Correcting shifts after the model has been deployed
Once a model is deployed, there are two main approaches to deal with data distribution shifts.

#### Retrain models periodically
This is, by far,  the most common strategy seen in industry.  In fact, it is so important that ["Chapter 9: Continual learning and testing in production"](09-continual-learning-and-test-in-production.md) is solely dedicated to this.

In this approach models get re-trained periodically (e.g. once a month, once a week, once a day).  There are 3 things to consider:
1. The decision on the optimal **retraining frequency** is important. However, many companies still determine it using gut feeling instead of experimental data. More about retraining frequency in  ["chapter 9 > how often to update your models".](09-continual-learning-and-test-in-production.md#How%20often%20to%20Update%20your%20models)
2. Retrain your model from scratch (aka stateless retraining) VS  continue training from the last checkpoint (aka stateful retraining, fine-tuning). More about this in [chapter 9 > stateless vs stateful training](09-continual-learning-and-test-in-production.md#Stateless%20retraining%20VS%20Stateful%20training) 
4. What data should you include in the retraining? e.g. last 24 hrs, last week, last 6 months, data from the point the shift started?

You will need to run experiments to decide on these 3 things.

#### Adapt a trained model to a target distribution without requiring new labels

Techniques in this approach vary in nature and the degree of adoption in research and industry. Two examples briefly mentioned in the book:
- Zhang et al (2013): causal interpretations together with kernel embedding of conditional and marginal distributions to correct models’ predictions for both covariate shifts and label shifts without using labels from the target distribution.
- Zhao et al (2020): an unsupervised domain adaption technique that can learn data representations that are invariant to changing distributions.


# Monitoring and Observability

Monitoring and Observability go hand in hand but strictly speaking they are slightly different things:
- **Monitoring:** is putting trackers, logs,  metrics, etc. in place to help us determine **when** something went wrong.
- **Observability:** refers to tools and setup that allows you to figure out **what** went wrong (i.e. observe the inner-workings of your system).

## Software related metrics
ML systems are also software systems, so all software observability practices apply. These are the type of things you want to track:
- Operational metrics:
	- Metrics of the network the system is running on: Network latency, Network load.
	- Machine health metrics: CPU/GPU utilization, memory utilization.
	- Application metrics: endpoint load, request success rate, endpoint latency.

Software systems also often use service level objectives (SLOs) or have service level agreements (SLAs) to ensure availability. When coming up with SLOs and SLAs *you* need to figure out what **"up"** means in *"the system must be 'up' 99.99% of the time"*. 
- For example , you may define **"up"** as median latency <= 200ms and p99 <=2 sec.  Then you measure the amount of time in a month that your system did not comply with this restriction to calculate your uptime percentage.

## ML-Specific metrics
Your system may be up and working, but if your predictions are garbage, you have a problem. This is where ML specific metrics come in.

Usually there are four things you want to monitor: 1. Accuracy,  2. Predictions, 3. Features and 4. Raw inputs.

There is an inherent tradeoff in monitoring at each of these 4 levels: 
- Higher level metrics like accuracy-related metrics are easier to understand and relate with business level metrics. However, they represent the output of a chain of complex transformations, so even if we know that something is wrong, we don't necessarily know why.
- Lower level metrics like raw input monitoring are far removed from the business and harder to setup. However, if a specific raw input metric is wrong, then you immediately know what the problem is.

![Metrics of different types of ML metrics](08-data-distribution-shifts-and%20monitoring-in-production.assets/ml-metrics-tradeoffs.png)


Another key and often overlooked part of ML observability is **model interpretability**. If your model's accuracy degrades or your predictions exhibit an anomaly, knowing how your model works and which features contribute the most to your predictions will help you a lot in identifying what went wrong and fixing it.
- This takes us back to the [interpretability vs performance trade-off](06-model-development-and-offline-evaluation.md#Tip%209%20Evaluate%20trade-offs) when selecting a model. More interpretable models are easier to monitor.

### Monitoring accuracy-related metrics
Setting up accuracy-related metrics is not always possible to do as it relies on your problem having [natural labels](04-training-data.md#Natural%20labels) (or some weaker proxy of a natural label). 

If your system receives **any type of user feedback for the predictions it makes** (click, hide, purchase, upvote, downvote, favorite, bookmark, share, etc) you **should definitely track it.**  Even if the feedback cannot be used to directly infer natural labels, it can be used to detect changes in your ML model's performance. Also keep in mind and track second order effects. For example, if the click-through rate of your recommendations stay the same, but your completion rate drops, that may be a sign that there is a problem.

If possible, engineer your systems so that they collect user's feedback. For example, add "up vote / down vote" or "not helpful" buttons. This can be used beyond accuracy related-metrics. For example, it can be used to inform which samples need to be sent to human annotation for future iterations.

\*Reminder: monitoring accuracy is the most powerful and practical way of monitoring [data distribution shifts.](08-data-distribution-shifts-and%20monitoring-in-production.md#Detection%20using%20accuracy-related%20metrics)


### Monitoring predictions

Prediction is the most common artefact that companies monitor. This is because they are easy to capture, easy to visualise and they have low-dimensionality. This latter attribute makes  their summary statistics straightforward to calculate end interpret.

**Monitor predictions for distribution shifts.**  If your model's weights have not changed, but the prediction distribution has, that generally indicates a change in the underlying input distribution.
- Since predictions are low dimensional, it is easy to compute two-sample tests to assess distribution changes.

**Monitor your predictions for anomalies.** If your predictions have rapid changes in behaviour, like suddenly predicting False for 10 mins straight, you may be having an ML incident.  Monitoring prediction for anomalies is much more instant than monitoring accuracy for anomalies as "natural labels" may take days to become available.


### Monitoring features
Compared to monitoring raw input data, feature monitoring is appealing because the features have predefined schemas and the information is usually in a more "workable state" (e.g. derived features from an image VS the actual image).

Things you can monitor in your features:
- Your features follow the expected schema.
	- The feature values satisfy a regular expression.
	- The feature values belong to a predefined set.
- The `min, max or median` of the feature are within acceptable ranges
- The value of feature A is always greater than feature B.

Two common libraries for doing this type of feature monitoring are Great Expectations and Deequ.

You can also use your feature monitoring to detect  [input data drift P(X)](08-data-distribution-shifts-and%20monitoring-in-production.md#Detection%20using%20statistical%20methods). If you plan to use statistical tests, you will need to do dimensionality reduction as features tend to be high-dimensional. However, dimensionality reduction reduces the effectiveness of statistical tests.

#### Challenges of feature monitoring
Feature monitoring is possible but it is also challenging. Here are some of the challenges you may encounter. Consider them so that you can select the level of feature monitoring that works for you:
1. You may have hundreds of models, each with hundreds of features. That is a lot of metrics to grok.
2. In practice, feature monitoring tends to be more useful for debugging purposes than for performance degradation detection. Adding automated drift alerts to all features will cause a lot of false positives.
3. Feature extraction may be a multi step and multi tool process (e.g. Snowflake > Pandas > Numpy). This makes it more difficult to choose what to monitor.
4. The feature schema may change overtime, requiring your expectation monitors to stay up to date.

### Monitoring raw inputs
In theory, monitoring raw inputs gives you the benefit of monitoring the "purest" version of your input and as such,  it should allow you to tell if your input distribution has really changed or a bug has been introduced downstream.

In practice, monitoring raw inputs is really tough and sometimes impossible:
- Raw inputs may be in formats that are very hard to work with: e.g. large assets, images / video / audio files in different formats, encrypted PII data.
- ML engineers might not even have access to the raw inputs for privacy reasons and they may be asked to query the data from a data warehouse in which the data is already partially processed.

For the reasons above, monitoring raw inputs usually falls under the data platform team's responsibility.

## Monitoring toolbox

From the implementation perspective, the pillars of monitoring are **metrics, logs and traces**. However, from the **"user monitoring the systems"** perspective the real pillars of monitoring are: **logs, dashboards and alerts.**

### Logs and distributed tracing
- If you have a distributed system (most likely you have), make sure your logs have *distributed tracing*.
- Record all event metadata with the logs: when it happens, what service it happens in, the function that was called, the user associated , etc. Log tagging is your friend for this.
- If you want to analyse your logs, analising billions of logs is futile. Companies use ML to do large scale analysis of logs.
- As a consumer of log technologies keep in mind that:
	- Log providers may process your logs periodically for certain attributes. This means that you can only discover certain problems periodically.
	- To discover things as soon as they happen, your log provider must be using stream processing technologies like Flink.

### Dashboards
- Dashboards show useful visualizations of metrics that are critical for monitoring.
- Dashboards makes monitoring accessible to non-engineers. Monitoring is not just for developers. Non-engineer stakeholders should also monitor their share of implications of having an ML product in production.
- Excessive metrics in a dashboard is counter productive. This is known as *dashboard rot.*
	- Be picky on the metrics you want in the dashboard and abstract out lower-level metrics by computing higher-level ones. 

### Alerts
- An alert is an automatic warning that is sent to a *notification channel* when a particular *alert policy* is violated.
- **Alert policy:** the condition that needs to be breached to trigger the alert and the severity associated to that breach.
- **Notification channels:** who needs to get notified? This typically is an email, a Slack channel and / or  an on-call roster.
- **Description of the alert:** make sure you point the person receiving the alert to a runbook that contains what to do.
- **Alert fatigue** is a real problem. 
	- Too many alerts will desensitize people and critical alerts will be ignores.
	- Only create alerts that are actionable
	- Only alert people out of office hours if the consequences of the alert condition being breached cannot wait until the next working day.