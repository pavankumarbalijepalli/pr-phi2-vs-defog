# Literature review: Fine-tuning Phi-2 on NL2SQL and comparing it with Defog SQL models:

## 1. Introduction

Abstract

## 2. Literature Review

### LLMs and Fine-tuning:

__LLMs: Large Language Models Explained__

Large Language Models (LLMs) are a recent innovation in natural language processing (NLP) that have captured the attention of researchers and the public alike. These powerful models are trained on massive amounts of text data, allowing them to learn complex relationships between words and generate human-quality text, translate languages, answer questions, and perform other creative tasks.

<img src="https://via.placeholder.com/640x360">

__Architecture:__

At the heart of most LLMs lie powerful neural network architectures known as Transformers. These models rely on attention mechanisms to process input text, allowing them to analyze the relationships between words within a sentence and across the entire text. Think of it as the LLM reading and rereading the text with a spotlight, focusing on different words and their connections to understand the overall meaning.

<img src="https://via.placeholder.com/640x360">

__Training:__

LLMs are trained on colossal datasets of text and code, often containing billions of words or more. This training involves unsupervised learning techniques like masked language modeling, where the LLM predicts missing words in masked sentences, and denoising autoencoders, where the model reconstructs corrupted text back to its original form. By going through these tasks countless times with diverse samples, the LLM learns the intricacies of language and builds a rich internal representation of the world.

<img src="https://via.placeholder.com/640x360">

__Capabilities:__

LLMs boast a wide range of capabilities, far exceeding basic language understanding. They can:
- __Generate text:__ From creative writing to code generation, LLMs can produce fluent and coherent text that often mimics human writing styles.
- __Translate languages:__ LLMs can accurately translate between languages, understanding the nuances of both and preserving the original meaning.
- __Answer questions:__ LLMs can process and answer open ended, challenging, or even strange questions by drawing on their vast knowledge base of text and code.
- __Summarize text:__ LLMs can condense lengthy documents into concise summaries, capturing the key points and discarding irrelevant details.
- __And much more:__ LLMs are constantly being explored and pushed to their limits, demonstrating new abilities like writing different kinds of creative content, composing music, and even writing computer programs.
However, it's important to remember that LLMs are still under development and can exhibit limitations like bias, factual errors, and difficulty handling complex reasoning tasks.

__Fine-tuning LLMs:__
Fine-tuning is a technique for adapting pre-trained LLMs like those described above to perform specific tasks. We can fine-tune an LLM to convert input text of questions into respective output text. This is a complex task that requires the LLM to understand both natural language and the specific syntax and rules of SQL databases.

Here's how fine-tuning works:

- __Select a dataset:__ We choose a dataset containing pairs of input text and their corresponding output text. This data will guide the LLM towards the specific task of fine-tuning.
- __Freeze most parameters:__ Instead of retraining the entire LLM from scratch, we "freeze" most of its parameters, preserving the general language knowledge from the original training data.
- __Fine-tune specific layers:__ We add or adjust a few additional layers specifically designed for the NL2SQL task. These layers are then trained on the chosen dataset, enabling the LLM to learn the mapping between natural language and SQL.
- __Evaluation and refinement:__ We evaluate the performance of the fine-tuned LLM on unseen data and make adjustments to the additional layers or training process as needed.

By fine-tuning, we leverage the general language understanding of the LLM while tailoring it to the specific requirements of NL2SQL. This can significantly improve the model's performance compared to training from scratch on limited NL2SQL data. Overall, LLMs represent a remarkable advancement in NLP, and fine-tuning unlocks their potential for specialized tasks like NL2SQL. We can expect further progress in both LLM design and fine-tuning techniques, enabling these powerful models to solve even more complex challenges and bridge the gap between natural language and computer code.

### NL2SQL:

NL2SQL, or Natural Language to SQL, is a subfield of NLP that focuses on translating natural language descriptions of queries into executable SQL statements. This essentially allows non-technical users to interact with databases using familiar language, democratizing data access and analysis.

__Applications of NL2SQL:__

- Business intelligence: NL2SQL empowers non-technical analysts to explore data and generate reports without writing complex SQL queries.
- Customer service: Chatbots can understand and respond to customer queries by directly accessing relevant information from databases.
- Education: Students can learn basic data analysis by formulating queries in natural language.

__Common Challenges in NL2SQL:__

- Complexity of natural language: Understanding the nuances of human language, including ambiguity, synonyms, and complex sentence structures, can be difficult for machines.
- Variety of database schemas: Adapting a model to different database structures and dialects of SQL presents a significant challenge.
- Handling complex queries: Translating queries involving aggregates, joins, and subqueries requires advanced reasoning and understanding of database relationships.
- Bias and fairness: NL2SQL models trained on biased data can perpetuate those biases in their outputs.

__Existing Approaches to NL2SQL:__

- Rule-based systems: These systems rely on hand-crafted rules to map natural language keywords to specific SQL clauses. They are often brittle and require significant manual effort for maintenance.
- Machine learning models: Techniques like neural networks and tree-based methods can learn the mapping from natural language to SQL from training data. These models offer more flexibility than rule-based systems but can require large amounts of data for good performance.
- LLM-based techniques: Large Language Models (LLMs) pre-trained on massive text datasets can be fine-tuned for NL2SQL tasks. Their advantage lies in their ability to capture complex linguistic relationships and adapt to new data quickly.

__Fine-tuning LLMs for NL2SQL Use Cases:__

- Reduced model size: By fine-tuning only a small portion of the LLM, we can create a significantly smaller model for NL2SQL, addressing your concern about accessibility for developers.
- Improved performance: LLMs offer the potential for better accuracy and robustness in handling complex queries compared to traditional models.
Faster development: Fine-tuning pre-trained LLMs requires less training data and can be quicker to implement than training new models from scratch.

---
## DONE TILL HERE
### Phi-2 and Defog SQL:
    * Describe the architecture and training of Phi-2 and Defog SQL.
    * Highlight their specific strengths and weaknesses for NL2SQL tasks, based on existing literature.

#### Fine-tuning Phi-2 for NL2SQL

* **Dataset Selection:**
    * Discuss the choice of dataset for fine-tuning Phi-2, considering size, diversity, and alignment with your evaluation tasks.
    * Explain the importance of dataset quality and potential biases.
* **Fine-tuning Techniques:**
    * Describe the specific fine-tuning approach used for Phi-2, including hyperparameter settings and optimization methods.
    * Reference relevant research on fine-tuning LLMs for NL2SQL tasks.
* **Evaluation Metrics:**
    * Define the metrics used to evaluate the performance of Phi-2 on NL2SQL tasks (e.g., accuracy, precision, recall, F1-score).
    * Explain the rationale behind the chosen metrics.

#### Comparison with Defog SQL Models

* **Review of Existing Comparisons:**
    * Summarize existing studies that compare Defog SQL models with other approaches for NL2SQL.
    * Extract key findings and insights relevant to your comparison.
* **Your Experimental Setup:**
    * Describe the dataset(s) and evaluation metrics used for your comparison.
    * Explain the configuration of Defog SQL models used for benchmarking.
* **Results and Analysis:**
    * Present the performance results of Phi-2 and Defog SQL models on the chosen evaluation tasks.
    * Conduct a thorough analysis of the results, identifying strengths and weaknesses of each model.
    * Discuss potential reasons for observed differences in performance.

---
**5. Discussion**

* **Key Findings and Implications:**
    * Summarize the main findings of your study, highlighting the implications for NL2SQL research and practice.
* **Open Challenges and Future Directions:**
    * Discuss challenges in fine-tuning Phi-2 for NL2SQL and comparing it with other models.
    * Identify potential future research directions, such as exploring different fine-tuning techniques, addressing model biases, or improving evaluation methods.

**6. Conclusion**

* **Reiterate key takeaways:**
    * Summarize the main contributions of your research and their significance.
* **Suggest future work:**
    * Propose potential avenues for further research in this area, building upon your findings.
