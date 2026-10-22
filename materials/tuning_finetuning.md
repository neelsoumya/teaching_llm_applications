# Tuning/finetuning and RLHF

- [🎥 Video CS395 Stanford Finetuning](https://youtu.be/PmW_TMQ3l0I?si=82nhWrdTQSbk0u7b)

- Model already knows what language is (after _pretraining_)

- model then tuned on specific tasks

- preference tuning

![image](../images/tuning_overview.png)


## Preference tuning

- collect preference pairs and train model on it

- We _prefer_ this 

- rather than say I want this

- easier to compare than generate from scratch

![image](../images/preference_tuning_overview.png)

- SFT data quality is very important

- 💡 If SFT data is _misbehaving_ then preference tuning may not be the answer: _check your SFT data_

- also see [RLHF material](week07_finetuning_and_rlhf.md)

- data collection

- pointwise vs. pairwise preference data

![image](../images/pointwise_vs_pairwise.png)

- collect _pairwise_ preference data for prompts

- LLM as a judge


## RLHF

- What is reinforcement learning?

![image](../images/reinformcement_learning_formulation.png)

