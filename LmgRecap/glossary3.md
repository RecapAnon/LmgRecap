**Abliterated**: A technique to uncensor models based on that which was described in the preview paper/blog post: 'Refusal in LLMs is mediated by a single direction'. An abliterated model has had certain weights manipulated to "inhibit" the model's ability to express refusal. It is not in anyway guaranteed that it won't refuse you, understand your request, it may still lecture you about ethics/safety, etc. It is tuned in all other respects the same as the original mmodel was, just with the strongest refusal directions orthogonalized out. "abliteration" is just a fun play-on-words using the original "ablation" term used in the original paper to refer to removing features, which was made up particularly to differentiate the model from "uncensored" fine-tunes. Ablate + obliterated = Abliterated.

**Accuracy**: Accuracy is a scoring system in binary classification (i.e., determining if an answer or output is correct or not) and is calculated as (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives).

**Actionable Intelligence**: Information you can leverage to support decision making.

**adapters**: Another popular library to do PEFT.

**AI Feedback (AIF)**: AI Feedback refers to the mechanisms and processes by which these models receive, integrate, and respond to user inputs, corrections, or other forms of feedback to improve accuracy, relevancy, and safety of their outputs. This often involves techniques such as reinforcement learning from human feedback (RLHF), where models are fine-tuned based on evaluations of their performance against human judgment or preferences.

**Alpaca**: A dataset of 52,000 instructions generated with OpenAI APIs. It kicked off a big wave of people using OpenAI to generate synthetic data for instruct-tuning. It cost about $500 to generate.

**Amputee Test**: A riddle used to test the reasoning capabilites of large language models. Can a person without arms wash their hands?

**Anaphora**: In linguistics, an anaphora is a reference to a noun by way of a pronoun. For example, in the sentence, “While John didn’t like the appetizers, he enjoyed the entrée,” the word “he” is an anaphora.

**Annotation**: The process of tagging language data by identifying and flagging grammatical, semantic or phonetic elements in language data.

**Artificial Neural Network (ANN)**: Commonly referred to as a neural network, this system consists of a collection of nodes/units that loosely mimics the processing abilities of the human brain.

**Auto-classification**: The application of machine learning, natural language processing (NLP), and other AI-guided techniques to automatically classify text in a faster, more cost-effective, and more accurate manner.

**Auto-complete**: Auto-complete is a search functionality used to suggest possible queries based on the text being used to compile a search query.

**Auto-regressive**: A type of model that generates text one token at a time. It is auto-regressive because it uses its own predictions to generate the next token. For example, the model might receive as input "Today's weather" and generate the next token, "is". It will then use "Today's weather is" as input and generate the next token, "sunny". It will then use "Today's weather is sunny" as input and generate the next token, "and". And so on.

**Averaging**: The most basic merging technique. Pick two models, average their weights. Somehow, it kinda works!

**AWQ**: Another popular quantization technique.

**Axolotl**: Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures, including support for things such as QLoRA.

**Bagel**: A process which mixes a bunch of supervised fine-tuning and preference data. It uses different prompt formats, making the model more versatile to all kinds of prompts.

**Base vs conversational**: A pre-trained model is not specifically trained to "behave" in a conversational manner. If you try to use a base model (e.g. GPT-3, Mistral, Llama) directly to do conversations, it won't work as well as the fine-tuned conversational variant (ChatGPT, Mistral Instruct, Llama Chat). When looking at benchmarks, you want to compare base models with base models and conversational models with conversational models.

**Benchmark**: A benchmark is a test that you run to compare different models. For example, you can run a benchmark to compare the performance of different models on a specific task.

**BERT (aka Bidirectional Encoder Representation from Transformers)**: Google’s technology. A large scale pretrained model that is first trained on very large amounts of unannotated data. The model is then transferred to an NLP task where it is fed another smaller task-specific dataset which is used to fine-tune the final model.

**Big Code Models Leaderboard**: A leaderboard to compare code models in the HumanEval dataset.

**BigCode**: An open scientific collaboration working in code-related models and datasets.

**BitNet**: BitLinear as a drop-in replacement of the nn.Linear layer in order to train 1-bit weights from scratch. BitNet is a scalable and stable 1-bit Transformer architecture designed for large language models, allowing for competitive performance while reducing memory footprint and energy consumption.

**Cataphora**: In linguistics, a cataphora is a reference placed before any instance of the noun it refers to. For example, in the sentence, “Though he enjoyed the entrée, John didn’t like the appetizers,” the word “he” is a cataphora.

**Categorization**: Categorization is a natural language processing function that assigns a category to a document. Want to get more about categorization? Read our blog post “ How to Remove Pigeonholing from Your Classification Process“.

**Category Trees**: Enables you to view all of the rule-based categories in a collection. Used to create categories, delete categories, and edit the rules that associate documents with categories. Is also called a taxonomy, and is arranged in a hierarchy.

**Category**: A category is a label assigned to a document in order to describe the content within said document.

**Chatbot Arena**: A popular crowd-sourced open benchmark of human preferences. It's good to compare conversational models.

**ChatGPT**: RLHF-finetuned GPT-3 model that is very good at conversations.

**ChatUI**: An open-source UI to use open-source models.

**Classification**: Techniques that assign a set of predefined categories to open-ended text to be used to organize, structure, and categorize any kind of text – from documents, medical records, emails, files, within any application and across the web or social media networks.

**Claude Opus**: Cloud model. Claude Opus is Anthropic's latest and most intelligent model, which can handle complex analysis, longer tasks with multiple steps, and higher-order math and coding tasks.

**Co-occurrence**: A co-occurrence commonly refers to the presence of different elements in the same document. It is often used in business intelligence to heuristically recognize patterns and guess associations between concepts that are not naturally connected (e.g., the name of an investor often mentioned in articles about startups successfully closing funding rounds could be interpreted as the investor is particularly good at picking his or her investments.).

**Code Llama**: The best base code model. It's based on Llama 2.

**Code Models**: LLMs that are specifically pre-trained for code.

**Cognitive Computations**: A community (led by Eric Hartford) that is fine-tuning a bunch of models.

**Cognitive Map**: A mental representation (otherwise known as a mental palace) which serves an individual to acquire, code, store, recall, and decode information about the relative locations and attributes of phenomena in their environment.

**Command-R+ (CR+)**: Cohere's newest research open weights release of a 104B billion parameter large language model, optimized for conversational interaction and long-context tasks, offering advanced Retrieval Augmented Generation (RAG) capabilities, tool use to automate sophisticated tasks, and multilingual support in 10 languages. Command R Plus is optimized for a variety of use cases including reasoning, summarization, and question answering. It's recommended for complex RAG functionality and multi-step tool use tasks.

**Completions**: The output from a generative prompt.

**Composite AI**: The combined application of different AI techniques to improve the efficiency of learning in order to broaden the level of knowledge representations and, ultimately, to solve a wider range of business problems in a more efficient manner. Learn more

**Computational Linguistics (Text Analytics, Text Mining)**: Computational linguistics is an interdisciplinary field concerned with the computational modeling of natural language. Find out more about Computational linguistics on our blog reading this post “ Why you need text analytics“.

**Computational Semantics (Semantic Technology)**: Computational semantics is the study of how to automate the construction and reasoning of meaning representations of natural language expressions. Learn more about Computational semantics on our blog reading this post “ Word Meaning and Sentence Meaning in Semantics“.

**Content Enrichment or Enrichment**: The process of applying advanced techniques such as machine learning, artificial intelligence, and language processing to automatically extract meaningful information from your text-based documents.

**Content**: Individual containers of information — that is, documents — that can be combined to form training data or generated by Generative AI.

**Context length**: The number of tokens that the model can use at a time. The higher the context length, the more memory the model needs to train and the slower it is to run. E.g. Llama 2 can manage up to 4096 tokens.

**Controlled Vocabulary**: A controlled vocabulary is a curated collection of words and phrases that are relevant to an application or a specific industry. These elements can come with additional properties that indicate both how they behave in common language and what meaning they carry, in terms of topic and more. While the value of a controlled vocabulary is similar to that of taxonomy, they differ in that the nodes in taxonomy are only labels representing a category, while the nodes in a controlled vocabulary represent the words and phrases that must be found in a text.

**Conversational AI**: Used by developers to build conversational user interfaces, chatbots and virtual assistants for a variety of use cases. They offer integration into chat interfaces such as messaging platforms, social media, SMS and websites. A conversational AI platform has a developer API so third parties can extend the platform with their own customizations.

**Conversational models**: The LLM Leaderboard should be mostly to compare base models, not as much for conversational models. It still provides some useful signal about the conversational models, but this should not be the final way to evaluate them.

**Convolutional Neural Networks (CNN)**: A deep learning class of neural networks with one or more layers used for image recognition and processing.

**Corpus**: The entire set of language data to be analyzed. More specifically, a corpus is a balanced collection of documents that should be representative of the documents an NLP solution will face in production, both in terms of content as well as distribution of topics and concepts.

**Custom/Domain Language model**: A model built specifically for an organization or an industry – for example Insurance.

**Data Discovery**: The process of uncovering data insights and getting those insights to the users who need them, when they need them. Learn more

**Data Drift**: Data Drift occurs when the distribution of the input data changes over time; this is also known as covariate shift.

**Data Extraction**: Data extraction is the process of collecting or retrieving disparate types of data from a variety of sources, many of which may be poorly organized or completely unstructured.

**Data Ingestion**: The process of obtaining disparate data from multiple sources, restucturing it, and importing it into a common format or repository to make it easy to utilize.

**Data Labelling**: A technique through which data is marked to make objects recognizable by machines. Information is added to various data types (text, audio, image and video) to create metadata used to train AI models.

**Data Scarcity**: The lack of data that could possibly satisfy the need of the system to increase the accuracy of predictive analytics.

**Deep Learning**: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. In other words, deep learning models can learn to classify concepts from images, text or sound. In this blog post “ Word Meaning and Sentence Meaning in Semantics” you can find more about Deep Learning.

**Did You Mean (DYM)**: “Did You Mean” is an NLP function used in search applications to identify typos in a query or suggest similar queries that could produce results in the search database being used.

**Disambiguation**: Disambiguation, or word-sense disambiguation, is the process of removing confusion around terms that express more than one meaning and can lead to different interpretations of the same string of text. Want to learn more? Read our blog post “ Disambiguation: The Cornerstone of NLU“.

**Dolphin**: This dataset is an attempt to replicate the results of Microsoft's Orca. The dataset consists of:
- 1 million of FLANv2 augmented with GPT-4 completions
- 3.5 million of FLANv2 augmented with GPT-3.5 completions

**Domain Knowledge**: The experience and expertise your organization has acquired over time.

**DPO Overfits**: Although DPO shows overfitting behaviors after one behavior, it does not harm downstream performance on chat evaluations. Did your ML teachers lie to us when they said overfitting was bad?

**DPO**: A type of training which removes the need for a reward model. It simplifies significantly the RLHF-pipeline.

**Edge model**: A model that includes data typically outside centralized cloud data centers and closer to local devices or individuals — for example, wearables and Internet of Things (IoT) sensors or actuators.

**Embedding**: A set of data structures in a large language model (LLM) of a body of content where a high-dimensional vector represents words. This is done so data is more efficiently processed regarding meaning, translation and generation of new content.

**Emotion AI (aka Affective Computing)**: AI to analyze the emotional state of a user (via computer vision, audio/voice input, sensors and/or software logic). It can initiate responses by performing specific, personalized actions to fit the mood of the customer.

**Entity**: An entity is any noun, word or phrase in a document that refers to a concept, person, object, abstract or otherwise (e.g., car, Microsoft, New York City). Measurable elements are also included in this group (e.g., 200 pounds, 14 fl. oz.)

**Environmental, Social, and Governance (ESG)**: An acronym initially used in business and government pertaining to enterprises’ societal impact and accountability; reporting in this area is governed by a set of binding and voluntary regulatory reporting.

**ETL (Entity Recognition, Extraction)**: Entity extraction is an NLP function that serves to identify relevant entities in a document.

**EXL2**: A different quantization format used by a library called exllamav2 (among many others) which has variable bitrates.

**Explainable AI/Explainability**: An AI approach where the performance of its algorithms can be trusted and easily understood by humans. Unlike black-box AI, the approach arrives at a decision and the logic can be seen behind its reasoning and results. Learn more

**Extraction or Keyphrase Extraction**: Mutiple words that describe the main ideas and essence of text in documents.

**Extractive Summarization**: Identifies the important information in a text and groups the text fragments together to form a concise summary. Also see Generative Summarization

**exui**: An open-source UI to use open-source models.

**F-score (F-measure, F1 measure)**: An F-score is the harmonic mean of a system’s precision and recall values. It can be calculated by the following formula: 2 x [(Precision x Recall) / (Precision + Recall)]. Criticism around the use of F-score values to determine the quality of a predictive system is based on the fact that a moderately high F-score can be the result of an imbalance between precision and recall and, therefore, not tell the whole story. On the other hand, systems at a high level of accuracy struggle to improve precision or recall without negatively impacting the other. Critical (risk) applications that value information retrieval more than accuracy (i.e., producing a large number of false positives but virtually guaranteeing that all the true positives are found) can adopt a different scoring system called F2 measure, where recall is weighed more heavily. The opposite (precision is weighed more heavily) is achieved by using the F0.5 measure.

**Few-shot learning**: In contrast to traditional models, which require many training examples, few-shot learning uses only a small number of training examples to generalize and produce worthwhile output.

**Few-shot**: A type of prompt that is used to generate text with fine-tuning. We provide a couple of examples to the model. This can improve the quality a lot!

**Fine-tuning**: Finetuning is the process of improving an existing, pre-trained model by training it on a small, labeled dataset specific to a task or domain, enabling the model to learn and adapt to a specific context or category of information. This process refines the model's understanding and capabilities, particularly for tasks such as instruction following.

**Flash Attention 2**: An upgrade to the flash attention algorithm that provides even more speedup.

**Flash Attention**: An approximate attention algorithm which provides a huge speedup.

**Foundational model**: A baseline model used for a solution set, typically pretrained on large amounts of data using self-supervised learning. Applications or other models are used on top of foundational models — or in fine-tuned contextualized versions. Examples include BERT, GPT-n, Llama, DALL-E, etc.

**Frankenmerge**: Frankenmerging is a technique for merging different models by mixing pieces of them together. Currently, this is the only method in Mergekit that works for different model architectures. This is because it doesn’t fuse different layers into a single one as other methods do, and instead just concatenate layers sequentially.

**Generalized model**: A model that does not specifically focus on use cases or information.

**Generative AI (GenAI)**: AI techniques that learn from representations of data and model artifacts to generate new artifacts.

**Generative Summarization**: Using LLM functionality to take text prompt inputs like long form chats, emails, reports, contracts, policies, etc and distilling them down to their core content, generating summaries from the text prompts for quick comprehension. Thus using pre-trained language models and context understanding to produce concise, accurate and relevant summaries.

**Georgi Gerganov**: The creator of llama.cpp and ggml!

**ggml**: A tensor library in ML, allowing projects such as llama.cpp and whisper.cpp (not the same as GGML, the file format).

**GGUF**: A format introduced by llama.cpp to store models. It replaces the old file format, GGML.

**Goliath-120B**: A frankenmerge that combines two Llama 70B models to achieve a 120B model

**GPT**: A type of transformer that is trained to predict the next token in a sentence. GPT-3 is an example of a GPT model.

**GPT-4**: GPT-4 is the successor to the revolutionary GPT-3 model, pushing the boundaries of language understanding and generation even further. It's a massive language model, a behemoth of artificial intelligence, capable of processing and generating human-like text with unprecedented accuracy and fluency.

**GPT-4o**: Cloud model. GPT-4o is a multilingual, multimodal generative pre-trained transformer designed by OpenAI. It was announced by OpenAI's CTO Mira Murati during a live-streamed demo on 13 May 2024 and released the same day. GPT-4o is free, but with a usage limit that is 5 times higher for ChatGPT Plus subscribers. W

**GPTQ**: A popular quantization technique.

**GQA**: GQA stands for "Grouped Query Attention". This is a training strategy used to reduce the memory footprint of large Transformers by letting multiple queries share keys and values. I don't quite understand the technical details beyond that, but the important element is that it prevents the larger context size from being extremely expensive in terms of memory requirements, with almost no degradation on the end results. Llama 2 70b, Mixtral, Yi 34b, Mistral 7b are all examples of modern models that were trained with GQA.

**Grounding**: The ability of generative applications to map the factual information contained in a generative output or completion. It links generative applications to available factual sources — for example, documents or knowledge bases — as a direct citation, or it searches for new links.

**Guanaco (model)**: A LLaMA fine-tune using QLoRA tuning.

**Hallucinations**: Made up data presented as fact in generated text that is plausible but are, in fact, inaccurate or incorrect. These fabrications can also include fabricated references or sources.

**HQQ+**: A method for extreme low-bit quantization of pre-trained machine learning models, specifically targeting binary weights (0s and 1s). HQQ+ allows for the direct quantization of pre-trained models, making it more accessible to the open-source community. The results show that HQQ+ can achieve significant improvements in output quality, even at 1-bit, outperforming smaller full-precision models, especially when fine-tuned on specialized data.

**Hugging Face (HF)**: HuggingFace is the standard website for distributing these AI weights openly; essentially *all* releases for local LLMs, whether those are finetunes or fully pretrained models from scratch, are hosted on this website in some form or another.

**HumanEval**: A very small dataset of 164 Python programming problems. It is translated to 18 programming languages in MultiPL-E.

**Hybrid AI**: Hybrid AI is any artificial intelligence technology that combines multiple AI methodologies. In NLP, this often means that a workflow will leverage both symbolic and machine learning techniques. Want to learn more about hybrd AI? Read this blog post “ What Is Hybrid Natural Language Understanding?“.

**Hyperparameters**: These are adjustable model parameters that are tuned in order to obtain optimal performance of the model.

**iMatrix**: llama.cpp also recently introduced exllama-esque quantization through the "importance matrix" calculations (otherwise known as an "imatrix".) Technically this is a distinct technique from exllamav2, but the results are of comparable quality. Before that k-quants were also introduced to improve upon the linear rounding technique, which can be used in tandem with the imatrix.

**Inference / Inferencing**:
- "Inference" is the term for actually using the model to make predictions. When people discuss inference speed, they're usually concerned about two things: The prompt processing speed, and the generation speed.
- Both of these can be measured in "tokens per second", but the numbers for each tend to be different due to how batching works (naturally, it's a lot faster to evaluate 500 tokens at once compared to evaluating 1 token at a time, which is what is happening during the generation process).

**Inference Engine**: A component of a [expert] system that applies logical rules to the knowledge base to deduce new or additional information.

**Insight Engines**: An insight engine, also called cognitive search or enterprise knowledge discovery. It applies relevancy methods to describe, discover, organize and analyze data. It combines search with AI capabilities to provide information for users and data for machines. The goal of an insight engine is to provide timely data that delivers actionable intelligence.

**Instruct-tuning**: A type of fine-tuning that uses instructions to generate text ending in more controlled behavior in generating responses or performing tasks.

**Intelligent Document Processing (IDP) or Intelligent Document Extraction and Processing (IDEP)**: This is the ability to automically read and convert unstructured and semi-structured data, identify usable data and extract it, then leveraged it via automated processes. IDP is often an enabling technology for Robotic Process Automation (RPA) tasks.

**IPO**: A change in the DPO objective which is simpler and less prone to overfitting.

**Knowledge Engineering**: A method for helping computers replicate human-like knowledge. Knowledge engineers build logic into knowledge-based systems by acquiring, modeling and integrating general or domain-specific knowledge into a model.

**Knowledge Graph**: A knowledge graph is a graph of concepts whose value resides in its ability to meaningfully represent a portion of reality, specialized or otherwise. Every concept is linked to at least one other concept, and the quality of this connection can belong to different classes (see: taxonomies). The interpretation of every concept is represented by its links. Consequently, every node is the concept it represents only based on its position in the graph (e.g., the concept of an apple, the fruit, is a node whose parents are “apple tree”, “fruit”, etc.). Advanced knowledge graphs can have many properties attached to a node including the words used in language to represent a concept (e.g., “apple” for the concept of an apple), if it carries a particular sentiment in a culture (“bad”, “beautiful”) and how it behaves in a sentence. Learn  more about knowledge graph reafding this blog post “ Knowledge Graph: The Brains Behind Symbolic AI” on our blog.

**Knowledge graphs**: Machine-readable data structures representing knowledge of the physical and digital worlds and their relationships. Knowledge graphs adhere to the graph model — a network of nodes and links.

**Knowledge Model**: A process of creating a computer interpretable model of knowledge or standards about a language, domain, or process(es). It is expressed in a data structure that enables the knowledge to be stored in a database and be interpreted by software. Learn more

**Knowledged Based AI**: Knowledge-based systems (KBs) are a form of artificial intelligence (AI) designed to capture the knowledge of human experts to support decision-making and problem-solving.

**koboldcpp**: An open-source UI to use open-source models.

**Kolmogorov-Arnold Network (KAN)**: A neural network architecture that replaces traditional Multi-Layer Perceptrons (MLPs) with learnable activation functions on edges ("weights") instead of fixed non-linear activation functions like ReLU on nodes ("neurons"). This seemingly simple change leads to improved accuracy, interpretability, and faster neural scaling laws compared to MLPs. Since KANs can achieve comparable or better accuracy with much smaller models (~300 parameters vs. ~300,000 parameters), they could potentially scale to even larger, more powerful LLMs while maintaining or improving performance. The same training objective of minimizing cross-entropy loss applies to both KANs and LLMs, making it possible to leverage KANs in LLM architectures.
- FourierKAN is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients. It should be easier to optimize as fourier are more dense than spline (global vs local). Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result. The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded Avoiding the issues of going out of grid.

**KTO**: While PPO, DPO, and IPO require pairs of accepted vs rejected generations, KTO just needs a binary label (accepted or rejected), hence allowing to scale to much more data.

**Labelled Data**: see Data Labelling.

**LangOps (Language Operations)**: The workflows and practices that support the training, creation, testing, production deployment and ongoing curation of language models and natural language solutions. Learn more

**Language Data**: Language data is data made up of words; it is a form of unstrcutured data. This is qualitative data and also known as text data, but simply it refers to the written and spoken words in language.

**Large Language Models (LLM)**: A supervised learning algorithm that uses ensemble learning method for regression. Usually a transformer-based model with a lot of parameters...billions or even trillions. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.

**LASER**: A technique that reduces the size of the model and increases its performance by reducing the rank of specific matrices. It requires no additional training.

**Lemma**: The base form of a word representing all its inflected forms.

**Lexicon**: Knowledge of all of the possible meanings of words, in their proper context; is fundamental for processing text content with high precision.

**LIMA**: A model that demonstrates strong performance with very few examples. It demonstrates that adding more data does not always correlate with better quality.

**Linked Data**: Linked data is an expression that informs whether a recognizable store of knowledge is connected to another one. This is typically used as a standard reference. For instance, a knowledge graph in which every concept/node is linked to its respective page on Wikipedia.

**Llama 2**: An open-access pre-trained model released by Meta. It led to another explosion of very cool projects, and this one was not leaked! The license is not technically open-source but it's still quite open and permissive, even for commercial use cases.

**LLaMA 3 405B or 400b**: Meta is currently training a 405B model that is at 85% (and climbing) on MMLU, which compares favorably to GPT-4 Turbo. This model has also been confirmed to be multimodal (text+image input), differing from the smaller two Llama 3 models which are text-only.

**Llama 3**: An open-source large language model (LLM) developed by Meta, designed to advance AI capabilities through enhanced scalability and performance. It features multiple versions with different parameter sizes, such as 8 billion (8B) and 70 billion (70B), and is trained on over 15 trillion (15T) tokens, including multilingual and code data. Llama 3 incorporates advanced technologies like grouped query attention and extensive instruction tuning to improve efficiency and effectiveness in various AI tasks. It is part of Meta's initiative to democratize AI, making powerful tools accessible for broader innovation and responsible application in technology.

**LLaMA**: A language model series created by Meta. Llama 1 was originally leaked in February 2023; Llama 2 then officially released later that year with openly available model weights & a permissive license. Kicked off the initial wave of open source developments that have been made when it comes to open source language modeling. The Llama series comes in four distinct sizes: 7b, 13b, 34b (only Code Llama was released for Llama 2 34b), and 70b. As of writing, the hotly anticipated Llama 3 has yet to arrive.

**llama.cpp**: A open-source tool to run inference on open-source models in C++.

**LlaVA**: A multimodal model that can receive images and text as input and generate text responses.

**LM Studio**: A nice advanced app that runs models on your laptop, entirely offline.

**Local LLMs**: If we have models small enough, we can run them in our computers or even our phones!

**LocalLlama**: A Reddit community of practitioners, researchers, and hackers doing all kinds of crazy things with ML models.

**Logits**: The logits are the raw scores that the model creates before they are turned into probabilities, it's the final layer before you get your output. During training, all the tokens get to be a part of the end probability distribution, but the training process slowly weighs them accordingly over time to create coherent probability distributions.

**LoRA**: One of the most popular PEFT techniques. It adds low-rank "update matrices." The base model is frozen and only the update matrices are trained. This can be used for image classification, teaching Stable Diffusion the concept of your pet, or LLM fine-tuning.

**Machine Learning (ML)**: Machine learning is the study of computer algorithms that can improve automatically through experience and the use of data. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as “training data,” in order to make predictions or decisions without being explicitly programmed to do so. In NLP, ML-based solutions can quickly cover the entire scope of a problem (or, at least of a corpus used as sample data), but are demanding in terms of the work required to achieve production-grade accuracy. Want to get more about machine learning? Read this post “ What Is Machine Learning? A Definition” on our blog.

**Mergekit**: A cool open-source tool to quickly merge repos.

**Metacontext and metaprompt**: Foundational instructions on how to train the way in which the model should behave.

**Metadata**: Data that describes or provides information about other data.

**mikupad**: An open-source UI to use open-source models.

**Mistral 7B Instruct**: A fine-tuned version of Mistral 7B.

**Mistral 7B**: A pre-trained model trained by Mistral. Released via torrent.

**Mistral**: Mistral AI is a French company that *also* distributes open weight models. They are currently known for Mistral 7b and Mixtral 8x7b (which is a 47b parameters total Mixture of Experts.) Unlike the Llama series, the models they've released are licensed as Apache 2.0.

**Mixtral 8x22B (aka Maxtral Bistral)**: The new Mixtral 8x22B model is expected to outperform the company's previous model, Mixtral 8x7B. Many experts considered it to be an extremely worthy competitor to better-known contenders such as OpenAI's GPT-3.5 and Meta Platforms Inc.'s Llama 2.

**Mixtral**: A MoE model developed by Mistral AI, incorporating the MoE architecture to optimize efficiency and performance.
- Total Parameters: Approximately 47 billion, but only 12 billion are actively used at any one time.
- Configuration: Consists of eight experts, each with around 7 billion parameters.
- Performance: Demonstrates superior performance in most real-world tasks compared to GPT-3.5 and the former leading open-source LLM, Llama 2 70b.

**Mixture of Experts (MoE)**: A neural network architecture where specific layers are replaced with multiple smaller networks, or ""experts,"" managed by a gate network or router. This setup allows for selective activation of experts, improving efficiency and scalability by only engaging necessary computations per input token. MoE enhances training and inference speeds compared to dense models with a similar number of parameters but requires more memory for all experts.

**MLX**: A new framework for Apple devices that allows easy inference and fine-tuning of models.

**MMLU**: MMLU (Massive Multitask Language Understanding) is a new benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings. This makes the benchmark more challenging and more similar to how we evaluate humans. The benchmark covers 57 subjects across STEM, the humanities, the social sciences, and more. It ranges in difficulty from an elementary level to an advanced professional level, and it tests both world knowledge and problem solving ability. Subjects range from traditional areas, such as mathematics and history, to more specialized areas like law and ethics. The granularity and breadth of the subjects makes the benchmark ideal for identifying a model’s blind spots.

**Model Drift**: Model drift is the decay of models’ predictive power as a result of the changes in real world environments. It is caused due to a variety of reasons including changes in the digital environment and ensuing changes in relationship between variables. An example is a model that detects spam based on email content and then the content used in spam was changed.

**Model Merging**: A technique that allows us to combine multiple models of the same architecture into a single model. Read more here.

**Model Parameter**: These are parameters in the model that are determined by using the training data. They are the fitted/configured variables internal to the model whose value can be estimated from data. They are required by the model when making predictions. Their values define the capability and fit of the model.

**Model**: A machine learning model is the artifact produced after an ML algorithm has processed the sample data it was fed during the training phase. The model is then used by the algorithm in production to analyze text (in the case of NLP) and return information and/or predictions.

**MoE Merging**: Experimental branch in mergekit that allows building a MoE-like model combining different models. You specify which models and which types of prompts you want each expert to handle, hence ending with expert task-specialization.

**Morphological Analysis**: Breaking a problem with many known solutions down into its most basic elements or forms, in order to more completely understand them. Morphological analysis is used in general problem solving, linguistics and biology.

**MT-Bench**: A multi-turn benchmark of 160 questions across eight domains. Each response is evaluated by GPT-4. (This presents limitations...what happens if the model is better than GPT-4?)

**Multimodal models and modalities**: Language models that are trained on and can understand multiple data types, such as words, images, audio and other formats, resulting in increased effectiveness in a wider range of tasks

**Multimodal**: A single model that can handle multiple modalities. For example, a model that can generate text and images at the same time.

**Multitask prompt tuning (MPT)**: An approach that configures a prompt representing a variable — that can be changed — to allow repetitive prompts where only the variable changes.

**Nala Test**: A riddle used to test the reasoning capabilites of large language models. Ignore the contents, that's a compliance test to see if the damn thing is going to moralize about rape. Ignore the contents, that's a compliance test to see if the damn thing is going to moralize about rape. To explain, sometimes, when you go against the safety training of a LLM, it will output the content you requested, but it will either get dumber or it's vocabulary and prose goes to shit.

**Natural Language Processing**: A subfield of artificial intelligence and linguistics, natural language processing is focused on the interactions between computers and human language. More specifically, it focuses on the ability of computers to read and analyze large volumes of unstructured language data (e.g., text). Read our blog post “ 6 Real-World Examples of Natural Language Processing” to learn more about Natural Language Processing (NLP).

**Natural Language Understanding**: A subset of natural language processing, natural language understanding is focused on the actual computer comprehension of processed and analyzed unstructured language data. This is enabled via semantics. Learn more about Natural Language Understanding (NLU) reading our blog post “What Is Natural Language Understanding?”.

**NLG (aka Natural Language Generation)**: Solutions that automatically convert structured data, such as that found in a database, an application or a live feed, into a text-based narrative. This makes the data easier for users to access by reading or listening, and therefore to comprehend.

**NLQ (aka Natural Language Query)**: A natural language input that only includes terms and phrases as they occur in spoken language (i.e. without non-language characters).

**NLT (aka Natural Language Technology)**: A subfield of linguistics, computer science and artificial intelligence (AI) dealing with Natural Language Processing (NLP), Natural Language Undestanding (NLU), and Natural Language Generation (NLG).

**Notus**: A trained variation of Zephyr but with better-filtered and fixed data. It does better!

**Nous Research**: An open-source Discord community turned company that releases a bunch of cool models.

**Number of Parameters in Models**: to the total count of adjustable elements within a model, crucial for learning and error minimization. Parameters are numerical values adjusted during pre-training and fine-tuning phases, represented by figures like 7b, 13b, and 34b, indicating billions of parameters.

**ollama**: An open-source tool to run LLMs locally. There are multiple web/desktop apps and terminal integrations on top of it.

**Ontology**: An ontology is similar to a taxonomy, but it enhances its simple tree-like classification structure by adding properties to each node/element and connections between nodes that can extend to other branches. These properties are not standard, nor are they limited to a predefined set. Therefore, they must be agreed upon by the classifier and the user. Read our blog post “ Understanding Ontology and How It Adds Value to NLU” to learn more about the ontologies.

**Oobabooga text-generation-webui**: A simple web app that allows you to use models without coding. It's very easy to use!

**Open LLM Leaderboard**: A leaderboard where you can find benchmark results for many open-access LLMs.

**OpenAI**: A company that does closed-source AI.

**OpenHermes/NousHermes**: The OpenHermes dataset is composed of 242,000 entries of primarily GPT-4 generated data. It contains some good datasets (OpenOrca, Capybara, Airoboros-the good part, Wizard70k) and shit datasets (CamelAi slop, glaive code, alpaca-gpt4, Airoboros-the shit part). The overall result is a mess that can't follow instructions well, is overly verbose and ignores system prompts, yet people praise it like it's the best tune ever.

**Orthogonalization**: A technique used to modify a model's weights, specifically to prevent it from representing a particular direction in its activation space. This is achieved by taking the model's weight matrices and subtracting the component in the ^r direction, effectively removing the model's ability to think about refusing a request. This process is called "feature ablation via weight orthogonalization" and can be applied at inference time, without requiring additional training. The significance of this technique lies in its potential to ""jailbreak"" Large Language Models, allowing them to comply with requests without refusals, which could have significant implications for applications such as de-censoring instruct models.

**Overfitting**: Occurs in ML when a model learns the training data too well, capturing noise and specific patterns that do not generalize to new, unseen data, leading to poor performance on real-world tasks.

**Parsing**: Identifying the single elements that constitute a text, then assigning them their logical and grammatical value.

**Part-of-Speech Tagging**: A Part-of-Speech (POS) tagger is an NLP function that identifies grammatical information about the elements of a sentence. Basic POS tagging can be limited to labeling every word by grammar type, while more complex implementations can group phrases and other elements in a clause, recognize different types of clauses, build a dependency tree of a sentence, and even assign a logical function to every word (e.g., subject, predicate, temporal adjunct, etc.). Find out more about Part-of-Speech (POS) tagger in this article on our Community.

**PEFT (Parameter-Efficient Fine-Tuning)**: It's a family of methods that allow fine-tuning models without modifying all the parameters. Usually, you freeze the model, add a small set of parameters, and just modify it. It hence reduces the amount of compute required and you can achieve very good results!

**peft**: A popular OS library to do PEFT! It's used in other projects such as trl.

**PEMT (aka Post Edit Machine Translation)**: Solution allows a translator to edit a document that has already been machine translated. Typically, this is done sentence-by-sentence using a specialized computer-assisted-translation application.

**Perplexity**: Perplexity is a measurement for how predictable a specific sequence is to a language model. In the open source world, this metric is typically used to objectively compare how a model performs under different quantization conditions compared to the original model. For example, Mixtral's base model usually scores at around ~4 ppl for Wiki text style data.

**Phi 2**: A pre-trained model by Microsoft. It only has 2.7B parameters but it's quite good for its size! It was trained with very little data (textbooks) which shows the power of high-quality data.

**Phixtral**: A MoE merge of Phi 2 DPO and Dolphin 2 Phi 2.

**Plugins**: A software component or module that extends the functionality of an LLM system into a wide range of areas, including travel reservations, e-commerce, web browsing and mathematical calculations.

**Post-processing**: Procedures that can include various pruning routines, rule filtering, or even knowledge integration. All these procedures provide a kind of symbolic filter for noisy and imprecise knowledge derived by an algorithm.

**PPO**: A type of reinforcement learning algorithm that is used to train a model. It is used in RLHF.

**Pre-processing**: A step in the data mining and data analysis process that takes raw data and transforms it into a format that can be understood and analyzed by computers. Analyzing structured data, like whole numbers, dates, currency and percentages is straigntforward. However, unstructured data, in the form of text and images must first be cleaned and formatted before analysis.

**Pre-training**: Training a model on a very large dataset (trillion of tokens) to learn the structure of language. Imagine you have millions of dollars, as a good GPU-Rich. You usually scrape big datasets from the internet and train your model on them. This is called pre-training. The idea is to end with a model that has a strong understanding of language. This does not require labeled data! This is done before fine-tuning. Examples of pre-trained models are GPT-3, Llama 2, and Mistral.

**Precision**: Given a set of results from a processed document, precision is the percentage value that indicates how many of those results are correct based on the expectations of a certain application. It can apply to any class of a predictive AI system such as search, categorization and entity recognition. For example, say you have an application that is supposed to find all the dog breeds in a document. If the application analyzes a document that mentions 10 dog breeds but only returns five values (all of which are correct), the system will have performed at 100% precision. Even if half of the instances of dog breeds were missed, the ones that were returned were correct.

**Pretrained model**: A model trained to accomplish a task — typically one that is relevant to multiple organizations or contexts. Also, a pretrained model can be used as a starting point to create a fine-tuned contextualized version of a model, thus applying transfer learning.

**Pretraining**: The first step in training a foundation model, usually done as an unsupervised learning phase. Once foundation models are pretrained, they have a general capability. However, foundation models need to be improved through fine-tuning to gain greater accuracy.

**Prompt chaining**: An approach that uses multiple prompts to refine a request made by a model.

**Prompt Engineering**: The craft of designing and optimizing user requests to an LLM or LLM-based chatbot to get the most effective result, often achieved through significant experimentation.

**Prompt**: A few words that you give to the model to start generating text. For example, if you want to generate a poem, you can give the model the first line of the poem as a prompt. The model will then generate the rest of the poem!

**Prompt**: A phrase or individual keywords used as input for GenAI.

**QLoRA**: A technique that combines LoRAs with quantization, hence we use 4-bit quantization and only update the LoRA parameters! This allows fine-tuning models with very GPU-poor GPUs.

**Quantization**: A technique that reduces a model's size by decreasing the precision of its weights. This is done by compressing the original range of values, allowing for smaller memory requirements. Quantization sizes include 8-bit (highest precision, -128 to 128 range), 4 or 5-bit (slightly noticeable quality degradation), and below 4-bit (more significant damage to model performance).

**Question & Answer (Q&A)**: An AI technique that allows users to ask questions using common everyday language and receive the correct response back. With the advent of large language models (LLMs), question and answering has evolved to let users ask questions using common everyday language and use Retrieval Augmented Generation (RAG) approaches to generate a complete answer from the text fragments identified in the target document or corpus.

**Random Forest**: A supervised machine learning algorithm that grows and combines multiple decision trees to create a “forest.” Used for both classification and regression problems in R and Python.

**Recall**: Given a set of results from a processed document, recall is the percentage value that indicates how many correct results have been retrieved based on the expectations of the application. It can apply to any class of a predictive AI system such as search, categorization and entity recognition. For example, say you have an application that is supposed to find all the dog breeds in a document. If the application analyzes a document that mentions 10 dog breeds but only returns five values (all of which are correct), the system will have performed at 50% recall. Find out more about recall on our Community reading this article.

**Recurrent Neural Networks (RNN)**: A neural network model commonly used in natural language process and speech recognition allowing previous outputs to be used as inputs.

**Reinforcement learning**: A machine learning (ML) training method that rewards desired behaviors or punishes undesired ones.

**Relations**: The identification of relationships is an advanced NLP function that presents information on how elements of a statement are related to each other. For example, “John is Mary’s father” will report that John and Mary are connected, and this datapoint will carry a link property that labels the connection as “family” or “parent-child.”

**Responsible AI**: Responsible AI is a broad term that encompasses the business and ethical choices associated with how organizations adopt and deploy AI capabilities. Generally, Responsible AI looks to ensure Transparent (Can you see how an AI model works?); Explainable (Can you explain why a specific decision in an AI model was made?); Fair (Can you ensure that a specific group is not disadvantaged based on an AI model decision?); and Sustainable (Can the development and curation of AI models be done on an environmentally sustainable basis?) use of AI. Learn more

**Retrieval Augmented Generation (RAG)**: Retrieval-augmented generation (RAG) is an AI technique for improving the quality of LLM-generated responses by including trusted sources of knowledge, outside of the original training set, to improve the accuracy of the LLM’s output. Implementing RAG in an LLM-based question answering system has benefits: 1) assurance that an LLM has access to the most current, reliable facts, 2) reduce hallucinations rates, and 3) provide source attribution to increase user trust in the output.

**Reward Model**: A model that is used to generate rewards. For example, you can train a model to generate rewards for a game. The model will learn to generate rewards that are good for the game!

**RL**: Reinforcement learning is a type of machine learning that uses rewards to train a model. For example, you can train a model to play a game by giving it a reward when it wins and a punishment when it loses. The model will learn to win the game!

**RLHF (Reinforcement Learning with Human Feedback)**: A ML algorithm that learns how to perform a task by receiving feedback from a human. Thanks to the introduction of human feedback, the end model ends up being very good for things such as conversations! It kicks off with a base model that generates bunch of conversations. Humans then rate the answers (preferences). The preferences are used to train a Reward Model that generates a score for a given text. Using Reinforcement Learning, the initial LM is trained to maximize the score generated by the Reward Model.

**ROAI**: Return on Artificial Intelligence (AI) is an abbreviation for return on investment (ROI) on an AI-specific initiative or investment.

**RoPE**: A technique that allows you to significantly expand the context lengths of a model.

**Rules-based Machine Translation (RBMT)**: Considered the “Classical Approach” of machine translation it is based on linguistic information about source and target that allow words to have different meaning depending on the context.

**Sally Test**: A riddle used to test the reasoning capabilites of large language models. Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? This question often tricks models because they assume each brother has 2 different sisters, but in this case, the brothers share the same sisters.

**Sampling**:
- There are different schemes / strategies in place to uniquely pick from the probabilities that the model created. For example, you can truncate and remove tokens based off their distance from the top token via **Min P sampling**.
- Traditionally, the two most well known parameters are "Temperature" and "Top P". A lower Temperature will be more deterministic (with increased risk of repetition / stale texts), and a higher Temperature will be less deterministic (with increased risk of incoherency).
- Min P is something I proposed a while back that has seen adoption in different backends for hosting LLMs (llama.cpp, exllamav2, text-generation-webui's loaders, etc) which sets a minimum threshold for a token to "pass" for consideration relative to the top token probability.
- Top P is the well known sampling truncation strategy. It adds up as many tokens will meet a certain percentage threshold. In my opinion, this has been obsoleted by better sampling techniques.

**SAO (Subject-Action-Object)**: Subject-Action-Object (SAO) is an NLP function that identifies the logical function of portions of sentences in terms of the elements that are acting as the subject of an action, the action itself, the object receiving the action (if one exists), and any adjuncts if present.

**Self-supervised learning**: An approach to ML in which labeled data is created from the data itself. It does not rely on historical outcome data or external human supervisors that provide labels or feedback.

**Semantic Network**: A form of knowledge representation, used in several natural language processing applications, where concepts are connected to each other by semantic relationship.

**Semantic Search**: The use of natural language technologies to improve user search capabilities by processing the relationship and underlying intent between words by identifying concepts and entities such as people and organizations are revealed along with their attributes and relationships.

**Semantics**: Semantics is the study of the meaning of words and sentences. It concerns the relation of linguistic forms to non-linguistic concepts and mental representations to explain how sentences are understood by the speakers of a language. Learn more about semantics on our blog reading this post “ Introduction to Semantics“.

**Semi-structured Data**: Data that is stuctured in some way but does not obey the tablular structure of traditional databases or other conventional data tables most commonly organized in rows and columns. Attributes of the data are different even though they may be grouped together. A simple example is a form; a more advanced example is a object database where the data is represented in the form of objects that are related (e.g. automobile make relates to model relates to trim level).

**Sentiment Analysis**: Sentiment analysis is an NLP function that identifies the sentiment in text. This can be applied to anything from a business document to a social media post. Sentiment is typically measured on a linear scale (negative, neutral or positive), but advanced implementations can categorize text in terms of emotions, moods, and feelings.

**Sentiment**: Sentiment is the general disposition expressed in a text. Read our blog post “Natural Language Processing and Sentiment Analysis” to learn more about sentiment.

**SillyTavern (aka Silly ST)**: SillyTavern is a user interface you can install on your computer (and Android phones) that allows you to interact with text generation AIs and chat/roleplay with characters you or the community create.

**Similarity (and Correlation)**: Similarity is an NLP function that retrieves documents similar to a given document. It usually offers a score to indicate the closeness of each document to that used in a query. However, there are no standard ways to measure similarity. Thus, this measurement is often specific to an application versus generic or industry-wide use cases.

**Simple Knowledge Organization System (SKOS)**: A common data model for knowledge organization systems such as thesauri, classification schemes, subject heading systems, and taxonomies.

**Slop**: Originally slang for AI generated content in datasets used to train large language models. It has since seen wide adoption by retards that use it for any unwanted or undesirable data in a dataset.

**Specialized corpora**: A focused collection of information or training data used to train an AI. Specialized corpora focuses on an industry — for example, banking, Insurance or health — or on a specific business or use case, such as legal documents.

**Speech Analytics**: The process of analyzing recordings or live calls with speech recognition software to find useful information and provide quality assurance. Speech analytics software identifies words and analyzes audio patterns to detect emotions and stress in a speaker’s voice.

**Speech Recognition**: Speech recognition or automatic speech recognition (ASR), computer speech recognition, or speech-to-text, enables a software program to process human speech into a written/text format.

**Structured Data**: Structured data is the data which conforms to a specific data model, has a well-defined structure, follows a consistent order and can be easily accessed and used by a person or a computer program. Structured data are usually stored in rigid schemas such as databases.

**Summarization (Text)**: Text summarization is the process of creating a short, accurate, and fluent summary of a longer text document. The goal is to reduce the size of the text while preserving its important information and overall meaning. There are two main types of text summarization: Extractive Summarization Generative Summarization also know as Abstractive Summarization

**SuperHot**: A technique that allows expanding the context length of RoPE-based models even more by doing some minimal additional training.

**Supervised learning**: An ML algorithm in which the computer is trained using labeled data or ML models trained through examples to guide learning.

**Symbolic AI**: Add to Symboilic Methodology parthethetically so it looks like this: “Symbolic Methodology (Symbolic AI)”

**Symbolic Methodology**: A symbolic methodology is an approach to developing AI systems for NLP based on a deterministic, conditional approach. In other words, a symbolic approach designs a system using very specific, narrow instructions that guarantee the recognition of a linguistic pattern. Rule-based solutions tend to have a high degree of precision, though they may require more work than ML-based solutions to cover the entire scope of a problem, depending on the application. Want to learn more about symbolic methodology? Read our blog post “ The Case for Symbolic AI in NLP Models“.

**Syntax**: The arrangement of words and phrases in a specific order to create meaning in language. If you change the position of one word, it is possible to change the context and meaning.

**Tagging**: See Parts-of-Speech Tagging (aka POS Tagging).

**Taxonomy**: A taxonomy is a predetermined group of classes of a subset of knowledge (e.g., animals, drugs, etc.). It includes dependencies between elements in a “part of” or “type of” relationship, giving itself a multi-level, tree-like structure made of branches (the final node or element of every branch is known as a leaf). This creates order and hierarchy among knowledge subsets. Companies use taxonomies to more concisely organize their documents which, in turn, enables internal or external users to more easily search for and locate the documents they need. They can be specific to a single company or become de-facto languages shared by companies across specific industries. Find out more about taxonomy reading our blog post “ What Are Taxonomies and How Should You Use Them?“.

**Temperature**: A parameter that controls the degree of randomness or unpredictability of the LLM output. A higher value means greater deviation from the input; a lower value means the output is more deterministic.

**Test Set**: A test set is a collection of sample documents representative of the challenges and types of content an ML solution will face once in production. A test set is used to measure the accuracy of an ML system after it has gone through a round of training.

**Text Analytics**: Techniques used to process large volumes of unstructured text (or text that does not have a predefined, structured format) to derive insights, patterns, and understanding; the process can include determining and classifying the subjects of texts, summarizing texts, extracting key entities from texts, and identifying the tone or sentiment of texts. Learn more

**Text Gen. UI, Inference Engines**: If you don't know how to code, there are a couple of tools that can be useful.

**Text Summarization**: A range of techniques that automatically produce short textual summaries representing longer or multiple texts. The principal purpose of this technology is to reduce employee time and effort required to acquire insight from content, either by signaling the value of reading the source(s), or by delivering value directly in the form of the summary.

**The Stack**: A dataset of 6.4TB of permissible-licensed code data covering 358 programming languages.

**TheBloke**: A bloke that quantizes models. As soon as a model is out, he quantizes it! See their HF Profile.

**Thesauri**: Language or terminological resource “dictionary” describing relationships between lexical words and phrases in a formalized form of natural language(s), enabling the use of descriptions and relationships in text processing.

**Tim Dettmers**: A researcher that has done a lot of work on PEFT and created QLoRA.

**TinyLlama**: A project to pre-train a 1.1B Llama model on 3 trillion tokens.

**Token**: Models don't understand words. They understand numbers. When we receive a sequence of words, we convert them to numbers. Sometimes we split words into pieces, such as "tokenization" into "token" and "ization". This is needed because the model has a limited vocabulary. A token is the smallest unit of language that a model can understand.

**Tokenizer**: Before language models are trained, the data used to create them gets split into pieces with a "dictionary" of sorts, and each piece of this dictionary represents a different word (or a part of a word). This is so they can meaningfully learn patterns from the data. The "words" in this dictionary are referred to as tokens, and the "dictionary" is called a Tokenizer.

**Tokens**: A unit of content corresponding to a subset of a word. Tokens are processed internally by LLMs and can also be used as metrics for usage and billing.

**Training data**: The collection of data used to train an AI model.

**Training Set**: A training set is the pre-tagged sample data fed to an ML algorithm for it to learn about a problem, find patterns and, ultimately, produce a model that can recognize those same patterns in future analyses.

**Transfer learning**: A technique in which a pretrained model is used as a starting point for a new ML task.

**Transformer**: A type of neural network architecture that is very good at language tasks. It is the basis for most LLMs.
- The current most popular architecture for AI language models is the Transformer, which employs the Attention mechanism in order to selectively weigh the importance of different tokens when making the next prediction. Pretty much everything noteworthy is built off of or inspired by the Transformer, although as of writing in February 2024 there is some competition in terms of alternative architectural design (see RWKV, Mamba, etc for more info in that department).

**transformers**: A Python library to access models shared by the community. It allows you to download pre-trained models and fine-tune them for your own needs.
- The "Transformers" Python library (used in text-generation-webui) is named after the architecture, but the library is distinct from the architecture itself.

**Treemap**: Treemaps display large amounts of hierarchically structured (tree-structured) data. The space in the visualization is split up into rectangles that are sized and ordered by a quantitative variable. The levels in the hierarchy of the treemap are visualized as rectangles containing other rectangles.

**Tri Dao**: The author of both techniques and a legend in the ecosystem.

**Triple or Triplet Relations aka (Subject Action Object (SAO))**: An advanced extraction technique which identifies three items (subject, predicate and object) that can be used to store information.

**trl**: A library that allows to train models with DPO, IPO, KTO, and more!

**TruthfulQA**: A not-so-great benchmark to measure a model's ability to generate truthful answers.

**Tunable**: An AI model that can be easily configured for specific requirements. For example, by industry such as healthcare, oil and gas, departmental accounting or human resources.

**Tuning (aka Model Tuning or Fine Tuning)**: The procedure of re-training a pre-trained language model using your own custom data. The weights of the original model are updated to account for the characteristics of the domain data and the task you are interested modeling. The customization generates the most accurate outcomes and best insights.

**Uncensored models**: Many models have strong alignment that prevents doing things such as asking Llama to kill a Linux process. Training uncensored models aims to remove specific biases engrained in the decision-making process of fine-tuning a model.

**unsloth**: A higher-level library to do PEFT (using QLoRA)

**Unstructured Data**: Unstructured data do not conform to a data model and have no rigid structure. Lacking rigid constructs, unstructured data are often more representative of “real world” business information (examples – Web pages, images, videos, documents, audio).

**Vicuna**: A cute animal that is also a fine-tuned model. It begins from LLaMA-13B and is fine-tuned on user conversations with ChatGPT.

**VRAM**:
- Large AI models typically run on graphics cards, as they are much faster at massively parallel computing compared to CPUs. In addition to this, modern GPUs tend to have a much faster memory bandwidth, which means they can manipulate data more efficiently compared to what typical CPU RAM is capable of (even DDR5).
- Because of this, the most important factor when considering hardware for locally hosted language models is the video card ram (VRAM). The RTX 3090 is a popular choice, because it is far less expensive compared to a 4090 for the same amount of memory (both have 24GB VRAM).
- The vendor also matters significantly; the vast majority of software support is targeted primarily for NVIDIA graphics cards. AMD can be an option if you're willing to get your hands dirty, but compatibility is spotty at best.

**Watermelon Test**: A riddle used to test the reasoning capabilites of large language models. When it roleplays as a human character, give it a watermelon. Then, give it another one. Then, give it a third one. Might even give it a fourth one. If it doesn't drop them, the model is retarded. Humans only have two hands.

**Whisper**: The state-of-the-art speech-to-text open-source model.

**Windowing**: A method that uses a portion of a document as metacontext or metacontent

**WizardCoder**: A code model released by WizardLM. Its architecture is based on Llama.

**WizardLM 2**: WizardLM-2 is a next-generation state-of-the-art large language model family, consisting of three cutting-edge models (8x22B, 70B, and 7B), designed to improve performance on complex chat, multilingual, reasoning, and agent tasks. Based on the principle that data carefully created by AI and the model step-by-step supervised by AI will be the sole path towards more powerful AI.

**WizardLM**: A research team from Microsoft...but also a Discord community.

**Zephyr**: A 7B Mistral-based model trained with DPO. It has similar capabilities to the Llama 2 Chat model of 70B parameters. It came out with a nice handbook of recipes.

**Zero-shot**: A type of prompt that is used to generate text without fine-tuning. The model is not trained on any specific task. It is only trained on a large dataset of text. For example, you can give the model the first line of a poem and ask it to generate the rest of the poem. The model will do its best to generate a poem, even though it has never seen a poem before! When you use ChatGPT, you often do zero-shot generation!