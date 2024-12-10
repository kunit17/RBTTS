# RBTTS

Embarking on my second project, albeit one that could prove to be too challenging, as it delves into uncharted territory for me and demands skills I’m still mastering. My goal is to produce a natural-sounding, Text-To-Speech model using the Transformer architecture. Learning by coding directly (compared to online courses) has accelerated my knowledge base, however, so at the minimum I expect to gain a lot of insight into Transformers, in general.

# Challenges and (semi-) completed steps

Pre-processing:
1) Gather suitable audio files and transcriptions (the normal route is to use publicly available data - I'll be using something more fun, however)
2) Create a tokenizer and encoder that uses padding as the input text I will be training on is variable in length
3) Express audio (wav) files as tensors using the Mel Scale
4) Ensure the tensors produced are correctly padded in such a way that the padding does not affect loss calculations

Model Coding:
1) TBC


![Text-to-speech](./Files/NN_wav.jpg)
