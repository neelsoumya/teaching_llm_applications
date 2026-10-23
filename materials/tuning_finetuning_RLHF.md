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

- _s_t_ is state you are in, _a_t_ is action you take, _r_t_ is reward you get

- sometimes a good or bad reward

- leverage this for _preference tuning_

- agent is the LLM

- state is prompt

- next token is next state

![image](../images/preference_reinforcement.png)

- tune the LLM

- sparse signals could be a problem in RL 

- Bradley-Terry model for preference learning

![image](../images/bradley_terry.png)

- 💡 binary cross entropy loss function since pairwise

- 💡 take _log_ of multiplication of probabilities: these numbers can get very small

![image](../images/reward_model_preference.png)

- 1000s of data points from humans

- _Model_ pretrained LLM with classification head instead of next token prediction (decoder only)

- is this regression or reinforcement learning?

- change weights of LLM using reinforcement learning

- reward model has been trained and frozen

![image](../images/idea_rlhf.png)

- do not deviate too much from base model: _catastrophic forgetting_