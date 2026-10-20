# Tuning/finetuning

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

- If SFT data is _misbehaving_ and preference tuning may not be the answer: _check your SFT data_