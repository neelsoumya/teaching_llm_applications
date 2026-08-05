# Softmax

$$softmax(x) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$   

- Intuition: turns logits into a probability distribution 

- Why we need it? To convert raw logits into probabilities that sum to 1. 

## 🤔 ❓ Question for class

- 🤔 ❓[What happens when you delete the first token and the relationship with softmax. Video by Vizuara on Youtube](https://www.youtube.com/shorts/nMU1-prYcO0)

- `softmax` learns to dump attention on first token

- if you remove that token what happens to the model? Does it change the subsequent token predictions?

- nowadays there is a dedicated `sink token` token used to improve training stability


