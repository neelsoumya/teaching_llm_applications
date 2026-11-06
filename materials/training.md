# LLM training

- [🎥 Stanford CME295 video](https://youtu.be/VlA_jt_3Qc4?si=ARV-6XNPgcjXC90c)


## Pretraining

- predict next token

![image](../images/pretraining.png)

- most expensive step

- _CommonCrawl_, github, stackoverflow

- size: 100B / 1T tokens

## FLOPS

- floating point operations

- 10^25 FLOPS

- _FLOPs = number of tokens * number of parameters_

- see [Resource accounting](materials/resource_accounting.md)


## Scaling

- [Scaling](materials/scaling.md)

- _Chinchilla law_


## LLM training

- decoder only transformer based models

- memory bottleneck

- data parallelism: data distributed on GPUs (each device has a 👮 copy of the model)

- gradient aggregate across GPUs

![image](../images/data_parallelims.png)

- communication cost between GPUs

## Redundancy

![image](../images/redundancy_sharing.png)

- can share/_shard_ parameters across devices

- more communication costs but decreased memory load per GPU

## Model parallelism

- one expert on one device, another expert on another device

## Flash Attention

- [Flash attention](materials/flash_attention.md)

![image](../images/hbm_flash.png)

