## Advancing the Landscape of NL2SQL: A Comprehensive Evaluation of Fine-tuned Phi-2 and DeFog sqlcoder Approaches

The ongoing evolution of Natural Language to SQL translation (NL2SQL) presents a critical intersection, accentuating the delicate balance between accuracy and usability <sup>[1]</sup>. This research thoroughly examines the leader of small language models, Phi-2 (Recently introduced by Microsoft)<sup>[2]</sup> and the leader in NL2SQL tasks, DeFog SQLCoder, to illuminate the future trajectory of NL2SQL.

While DeFog SQLcoder excels in generating accurate SQL queries with 77.5% accurate generation, it is incapable of running on lower-computing machines. While SQLCoder has multiple models with varying parameter sizes, the smallest one with 7 Billion Parameters, requires a High RAM T4 GPU to infer<sup>[3]</sup>. In contrast, Phi-2 is a small language model explicitly designed to run on smaller machines<sup>[2]</sup> with performance better than most of the Large Language Models<sup>[4]</sup>. This study aims to explore and compare their performance across diverse NL2SQL tasks, introducing a nuanced understanding of their strengths and limitations.

Three pivotal questions guide our investigation:
- **Accuracy Parity**: Can Phi-2, through fine-tuning, achieve accuracy levels, i.e., 77.5% on the largest model, comparable to those of DeFog SQLCoder across a variety of NL2SQL tasks?
- **Interpretability Enhancement**: Does the fine-tuning of Phi-2 with an SQL dataset result in improved interpretability and explainability of generated queries when compared to DeFog models?
- **Difficulty Level Dynamics**: How do the performance and interpretability of Phi-2 and DeFog vary across difficulty levels, ranging from Easy to Medium and Hard?

Our comparative analysis aims to bridge the accuracy-deployability gap inherent in NL2SQL models, shedding light on the unique attributes of both Phi-2 and DeFog approaches. By examining their performance under different difficulty levels, we seek to provide a nuanced understanding of the strengths and weaknesses inherent in each model.

This research contributes to the academic discourse surrounding NL2SQL and holds practical implications for developers and users. The insights gained from this study have the potential to reshape the future of NL2SQL development. By prioritising deployability alongside accuracy, we aim to pave the way for capable yet smaller models. This, in turn, empowers developers and users to easily interact with and comprehend the logic behind generated SQL queries without using costly cloud-based compute instances.

Through this fine-tuning and comparative analysis, we anticipate contributing to the ongoing dialogue on the intersection of accuracy and deployability in NL2SQL, aiming to advance the field towards more feasible and CPU-friendly solutions.

**Keywords:** NL2SQL, Phi-2, DeFog, Deployability, Fine-tuning, Accuracy, Difficulty Levels, Comparative Analysis.

**References:**
- [1] - Xu, Canwen, et al. "Small models are valuable plug-ins for large language models." arXiv preprint arXiv:2305.08848 (2023).
- [2] - Li, Yuanzhi, et al. "Textbooks are all you need ii: phi-1.5 technical report." arXiv preprint arXiv:2309.05463 (2023).
- [3] - Korthikanti, Vijay Anand, et al. "Reducing activation recomputation in large transformer models." Proceedings of Machine Learning and Systems 5 (2023).
- [4] - Microsoft Research. "Phi-2: The Surprising Power of Small Language Models." Microsoft Research Blog, 26 Oct. 2023

**Resources:**
- https://huggingface.co/microsoft/phi-2/discussions/19
- https://huggingface.co/microsoft/phi-2/discussions/25

**From Subramanian Sir**
- What are you going to improve, usability or performance or interpretebility?
- have a feasible, flexible, straight forward objective. 
- Perform good literature survey - 4 to 5 Papers doing similar approach
- For Midterm - Architecture, Flowdiagram, Tech stack, Approach (comparision between existing approaches)
- Evaluation Methods - evaluation between the approaches, evaluation between the training and testing.
