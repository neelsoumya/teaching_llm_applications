# Practical based on `Baby steps` paper

- Baby steps in evaluating the capacities of large language models by Michael C. Frank

> Even for highly simplified stimuli, some superficial stimulus features are often still confounded. with the manipulation of interest. Thus, the trick of a truly clever experimental design is to hold every aspect of the probe stimulus constant across conditions, while making a single key modification that changes the observer’s interpretation. Classic language learning experiments demonstrate this design by using the same probe stimulus (for example, the novel word ‘golatu’) but creating learning environments in which the statistics of its use differ (for example, one where the syllables ‘go’, ‘la’, and ‘bu’ follow each other consistently vs. one where they are heard together only via the conjunction of two other words, ‘pigola’ and ‘tudaro’) [7]. This kind of design ensures that prior associations do not bias the result; without a closely matched control condition, an incidental preference for the word ‘golatu’ might lead to the appearance of success even in the absence of learning. In the case of LLMs, such matched controls are especially critical because models encode a massive set of prior associations that could bias their responses.

- Have a trained GPT-2 model. Test it at inference time on text like `golatu`. Can it predict the next word?

- Out of distribution

- Generalization problem