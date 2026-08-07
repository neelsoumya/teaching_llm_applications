# Sarvam

- Need for sovereign models

- 📝 Reading on AI sovereignty

- 📝 [`AI Sovereignty in the Global South: Power, Dependency, and Strategic Futures` Simon Davies, Vikranth Harthikote Nagaraja, Innocent Nyalala, Nirav Bhatt, Soumya Banerjee Empowering Global South AI Workshop: The 40th Annual AAAI Conference on Artificial Intelligence 2026](https://github.com/neelsoumya/paper_preprints/blob/master/AAAI_2026_globalsouthAI.pdf)
and [here](https://www.researchgate.net/publication/398869310_AI_Sovereignty_in_the_Global_South_Power_Dependency_and_Strategic_Futures)

- 🎥📝 Listen to talk [here](https://www.youtube.com/watch?v=GXBwAT0MaUE)




- How were Sarvam models built?

- Initially on top of `Llama 2` architecture

- Key difference: vocabulary and training data

- [🎥 Video of Sarvam](https://www.youtube.com/shorts/RFxYQl8m1_c)

- Trained on a diverse corpus of Indian languages, including Hindi, Tamil, Telugu, and Bengali.

- **Improved tokenization for Indian languages**
    - Standard tokenizers split Indian language words into smaller units, which reduces model efficiency.
    - Sarvam developed a custom tokenizer that can handle Indian language words more effectively.

- **Fine-tuned for downstream tasks**
    - Sarvam fine-tuned its models on various downstream tasks, including question answering, text summarization, and sentiment analysis.
    - These models are publicly available on Hugging Face.

- **Open-source contribution**
    - Sarvam released its models under the Apache 2.0 license, allowing researchers and developers to use them freely.

- now pretraining from _scratch_ on Hopper series GPUs

- they also contribute to `Llama` open source repositories