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

### Phi-2 and Defog SQL:

Both Phi-2 and Defog SQLCoder 7b are powerful models tackling the NL2SQL challenge, but they take different approaches. Let's delve into their architecture, training, strengths, and weaknesses:

- __Phi-2:__
   - Architecture:
      - Small Transformer model: Built with fewer parameters than other LLMs, making it lighter and faster to train and run.
      - Dense Attention Network (DAN): Focuses on interactions between words in a sentence, capturing long-range dependencies crucial for understanding complex language.
      - Encoder-Decoder structure: Encodes the natural language query and decodes it into an executable SQL statement.
   - Training:
      - Supervised learning: Trained on a large dataset of paired natural language queries and their corresponding SQL statements.
      - Focus on general language understanding: Leverages pre-training on diverse text data before fine-tuning for NL2SQL.
   - Strengths:
      - Smaller size: More accessible for deployment and faster inference compared to larger LLMs.
      - Strong language understanding: Captures complex linguistic relationships in natural language queries.
      - Good performance on simple and intermediate queries: Demonstrates accuracy on a variety of NL2SQL tasks.
   - Weaknesses:
      - Limited capacity for complex queries: May struggle with tasks involving joins, aggregations, or subqueries.
      - Explainability challenges: Understanding how Phi-2 arrives at its SQL translations can be difficult.
      - Susceptibility to biases: Potential for inheriting biases from the pre-training data.
- __Defog SQLCoder 7b:__

   - Architecture:
      - Hybrid model: Combines a Syntactic Transformer Encoder with a Semantic Role Labeler Decoder.
      - Syntactic Transformer Encoder: Analyzes the syntax and structure of the natural language query.
      - Semantic Role Labeler Decoder: Identifies the semantic roles of words and constructs the corresponding SQL statement based on these roles.
   - Training:
      - Multi-stage training: First trained on synthetic SQL-like data, then fine-tuned on real-world NL2SQL pairs.
      - Focus on semantic roles: Emphasizes understanding the meaning and relationships between words in the query.
   - Strengths:
      - Stronger performance on complex queries: Handles joins, aggregations, and subqueries with greater accuracy than Phi-2.
      - Improved explainability: Semantic role labeling provides transparency into the model's reasoning process.
      - Less susceptible to biases: Multi-stage training with synthetic data helps mitigate bias issues.
   - Weaknesses:
      - Larger model size: Requires more computational resources for training and deployment compared to Phi-2.
      - Potentially slower inference: Decoding based on semantic roles might be computationally expensive.
      - Limited general language understanding: Focus on semantic roles could miss subtle nuances in natural language queries.

Choosing between Phi-2 and Defog SQLCoder 7b depends on your specific needs. If you require a smaller, faster model for simple and intermediate NL2SQL tasks, Phi-2 might be a good choice. However, if you need to handle complex queries with high accuracy and explainability, Defog SQLCoder 7b might be the better option, despite its larger size and potentially slower inference. Remember, this is just a high-level comparison based on existing literature. Further research and experimentation might be needed to determine the optimal model for your specific use case.

### Fine-tuning Phi-2 for NL2SQL

**Dataset Selection for Fine-tuning Phi-2: Size, Diversity, and Bias Considerations**

Dataset selection is crucial for maximizing the effectiveness of your fine-tuned Phi-2 model for NL2SQL tasks. Here's a breakdown of key considerations:

   - __Size:__
      - Larger datasets generally lead to better performance: More data exposes Phi-2 to a wider range of query styles and SQL constructs, enhancing its ability to generalize and handle unseen data.
      - Balance size with computational resources: Training on massive datasets can be computationally expensive, requiring specialized hardware and expertise. Consider your available resources and find a balance between data size and feasibility.
      - Tailor size to your evaluation tasks: If your primary concerns are simple and intermediate queries, a smaller dataset focused on those types might be sufficient.
      
   - __Diversity:__
      - Include diverse query styles and complexities: Choose a dataset representing a variety of natural language formulations, database schemas, and query types (e.g., selection, aggregation, joins). This broadens Phi-2's understanding and improves its adaptability to real-world scenarios.
      - Cover diverse database domains: If your NL2SQL application targets specific domain areas (e.g., finance, healthcare), prioritize datasets containing queries and databases relevant to those domains.
      - Beware of domain mismatch: Avoid heavily relying on a dataset specific to a single domain if your goal is broader applicability. This can lead to overfitting and poor performance on diverse tasks.
        
   - __Alignment with Evaluation Tasks:__
      - Match the types of queries in your evaluation dataset with the training data: This ensures that Phi-2 is trained on similar tasks it will be evaluated on, leading to more accurate and meaningful results.
      - Consider the evaluation metrics you will use: If your metrics focus on specific aspects of NL2SQL accuracy (e.g., handling joins, accuracy on complex queries), prioritize datasets providing data for those aspects.
      - Don't rely solely on benchmark datasets: While popular benchmarks like Spider and WikiSQL are valuable, consider supplementing them with domain-specific or task-specific data for tailored performance.
        
   - __Dataset Quality and Potential Biases:__
      - Seek high-quality, well-annotated datasets: Ensure the data is accurate, consistent, and free from errors or inconsistencies that could mislead the model.
      - Pay attention to potential biases: Datasets can reflect biases present in their source data or human annotations. Analyze the dataset for potential biases in wording, database representations, or query types, and take steps to mitigate them during training and evaluation.
      - Consider data augmentation techniques: Augmenting your dataset with synthetic data or paraphrased queries can further diversify it and help mitigate biases while improving generalizability.
   
**Fine-tuning Phi-2 for NL2SQL: Techniques and Resources**

Fine-tuning Phi-2 for NL2SQL requires careful consideration of hyperparameters and optimization methods to maximize its performance. Here's a breakdown of common approaches and relevant research:

   - __Fine-tuning Approach:__
      - Freeze most parameters: Unlike traditional training from scratch, fine-tuning typically freezes a large portion of the pre-trained LLM parameters (e.g., encoder layers). This preserves the general language understanding while allowing adjustments for the specific NL2SQL task.
      - Fine-tune decoder and additional layers: A smaller set of parameters in the decoder and potentially add-on layers are fine-tuned on the NL2SQL dataset. These layers learn the mapping between natural language and SQL syntax.
      - Loss function: Choosing an appropriate loss function is crucial. Options include:
         - Supervised learning loss: Measures the difference between the model's predicted SQL and the ground truth SQL (e.g., cross-entropy loss).
         - Reinforcement learning loss: Rewards the model for generating valid and semantically correct SQL statements.
              
   - __Hyperparameter Settings:__
      - Learning rate: This controls the pace of parameter updates during fine-tuning. Start with a smaller learning rate (e.g., 1e-5) than pre-training and adjust based on validation performance.
      - Optimizer: Popular choices include Adam and AdamW, known for their efficiency and stability in handling sparse gradients from LLMs.
      - Batch size: Larger batch sizes generally lead to faster training but require more memory. Tune the batch size based on your available resources and model configuration.
      - Gradient clipping: Prevents exploding gradients during training, especially with larger learning rates. Experiment with different clipping values to ensure stable training.
        
   - __Optimization Methods:__
      - Early stopping and checkpointing: Regularly monitor validation performance and stop training once it starts to plateau. Checkpointing allows you to resume training from a previous state if needed.
      - Warm-up schedule: Gradually increase the learning rate from a very low value over the first few training steps to stabilize the model before full learning.
      - Regularization techniques: L1 or L2 regularization can help prevent overfitting and improve generalizability.

**Evaluating Fine-tuned Phi-2: Choosing the Right Metrics for NL2SQL**

Evaluating the performance of your fine-tuned Phi-2 for NL2SQL requires choosing the right metrics to objectively assess its strengths and weaknesses. Here's a breakdown of essential metrics and their rationale:

   - __Accuracy and Accuracy@k:__
      - Definition: Measures the percentage of queries for which the generated SQL statement exactly matches the ground truth. Accuracy@k considers the top k predictions, allowing for partial credit when the exact match is not found.
      - Rationale: A basic and easy-to-understand metric, accuracy tells you how often Phi-2 gets the SQL syntax completely right. However, it can be overly strict and penalize minor differences that don't affect the semantic correctness of the query.
   
   - __Precision and Recall:__
      - Definition: Precision measures the proportion of generated SQL statements that are semantically correct (true positives), while recall measures the proportion of ground truth SQL statements that are successfully generated (true positives / all positives).
      - Rationale: These metrics provide a finer-grained picture of Phi-2's ability to capture the meaning of the query and translate it into a functionally correct SQL statement, even if the syntax might differ slightly.
   
   - __F1-score:__
      - Definition: Harmonic mean of precision and recall, providing a balanced measure of both aspects.
      - Rationale: F1-score offers a compromise between precision and recall, balancing strictness with coverage. It's a valuable metric when both aspects are equally important in your application.
   
   - __Additional Metrics:__
      - Semantic similarity metrics: BLEU and ROUGE score the semantic similarity between generated and ground truth SQL, offering a more nuanced evaluation of meaning preservation.
      - Execution time: Measures the time taken for the generated SQL to execute in the database, relevant for practical applications where efficiency matters.
      - Domain-specific metrics: Depending on your NL2SQL application's domain (e.g., finance, healthcare), consider specialized metrics that assess performance for particular types of queries or data.

## 3. Methodology

   - __Data and Datasets:__
      - The dataset being used here is the Hugging Face's b-mc2/sql-create-context. This combines both the Spider and WikiSQL datasets with additional context through the database schema related information.
      - The dataset has the columns: question, answer, context. This provides so much feasibility for the developers to create their own context based prompts based on the pre-trained models.
      - The usage of this model for training helps during the inference. The users just need to provide the relevant db schema to the model along with the question which can generate SQL. 

   - __Fine-tuning Phi-2__
      - By splitting the dataset of 78k rows into Train (74.1k), Validation (2.34k), Test (1.56k) datasets, we'll prepare the dataset.
      - With the dataset columns, we will prepare a instruct prompt for the Phi-2 which will be used for fine-tuning.
      - Configuring BitsAndBytes for quantization is a crucial step and leads to easier and efficient fine-tuning.
      - Configuring Training Arguments / Hyperparameters with relevant values.
      - Configuring Lora parameters is also a crucial step, as it will ensure parameter efficient fine-tuning.
      - After this we can instantiate the Trainer Module from transformers, and start fine-tuning.
      - Must consider the possibility of mismatched versions or gpu related issues during fine-tuning and resolve them.
      - At each step, we should fine-tune over validation dataset, to understand the state of configurations.

   - __Comparison with DeFoG SQLCoder__ 
      - Once fine-tuning is completed, we should store the model in local or hf_hub.
      - In a new kernel, using the test dataset, we shall infer on the fine-tuned phi-2 model, and store various factors of result.
      - In another new kernel, using the test dataset, we shall infer on the defog SQLCoder-7b model, and store various factors of result.
      - Storing the scores of Phi-2 and Defog SQLCoder and saving them in another csv file helps in evaluation.

   - __Evaluation Metrics__
      - Once the csv file is created, we load the scoring dataframe.
      - From the columns of scoring dataframe we will fetch the generated SQL and actual SQL and compare them.
      - The inference times, gpu usage per inference, overall gpu usage, cpu usage per inference, overall cpu usage will also be considered to establish comparision between Phi-2 and Defog SQLCoder.

   - __Analysis and Findings__   
      - The findings shall be documented and a deck shall be prepared to showcase the detailed analysis.
      - The comparision will also showcase if the fine-tuned Phi-2 is better than Defog SQLCoder.


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


__References:__
- "Fine-tuning Language Models for NL2SQL with Noisy Data" by Chen et al. (2023): Explores fine-tuning LLMs for NL2SQL on noisy and incomplete datasets, addressing a common challenge in real-world applications.
- "Learning to Generate SQL Query Plans from Natural Language Descriptions" by Yin et al. (2022): Introduces a novel approach for NL2SQL that generates query plans alongside SQL statements, providing more flexibility and efficiency.
- "A Survey of Neural Natural Language to SQL Parsing" by Hu et al. (2021): Offers a comprehensive overview of various neural LLM-based approaches for NL2SQL, including their strengths and weaknesses.

