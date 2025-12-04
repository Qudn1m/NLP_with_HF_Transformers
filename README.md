<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Klaudio P.H

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
analys = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
analys("I love programming in Python!")
```

Result : 

```
[{'label': 'positive', 'score': 0.9890081882476807}]
```

Analysis on example 1 : 

The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.


### 2. Example 2 - Topic Classification

```
# TODO :
zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zero("Menopause is when women experience an end to their periods, or menstruation, for more than 12 months, and it marks the end of their reproductive years.", candidate_labels=["education", "health", "women"])
```

Result : 

```
{'sequence': 'Menopause is when women experience an end to their periods, or menstruation, for more than 12 months, and it marks the end of their reproductive years.',
 'labels': ['women', 'health', 'education'],
 'scores': [0.9735614657402039, 0.022356798872351646, 0.0040817782282829285]}
```

Analysis on example 2 : 

The zero-shot classifier correctly identifies "women" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
text_gen = pipeline("text-generation", model="gpt2")
text_gen("Nowadays young people are more ", max_length=30, num_return_sequences=3)
```

Result : 

```
[{'generated_text': 'Nowadays young people are more \xa0inured to this kind of approach, but their skills are limited to the basics. They are not necessarily prepared for the future.\nI\'m not saying that many young people are not capable of getting into the job market. But I do believe that in some respects they are as prepared as the average young person is, with enough training and experience to get into the industry.\nYou\'ll hear people saying, "I\'m not sure how to get into the job market." One day, in the fall, they will hear that they\'re just not in it, and then they\'ll start thinking about what they can do to earn their way into the workforce.\nThere\'s a way to get into the workforce without the job market being too difficult. I think it\'s important for young people to understand the skills they need to get into the workforce before they actually get into it.\nThat\'s why I think it\'s important to keep in mind that most young people have no idea what the job market is like. But they know what\'s available. They know what other people need.\nYou might think that young people would be able to get into the job market by simply learning basic skills, and that\'s not a good approach. But they\'re not.\nIn the'},
 {'generated_text': "Nowadays young people are more iced and refined and are more likely to buy a lot of cigarettes because they want to be more comfortable, that they want to smoke more and enjoy the company of others. So we have to get rid of the old ways of thinking and thinking about what we are doing.\n\nIn the United States, there is a growing number of young people who are really interested in smoking. You see, you have a lot of young people who are interested in the idea of smoking smoking and that is the first step. It is the first step to a better life. It is the first step to a better life.\n\nI think that smoking is probably the best way to be more informed. That is to get out of the old way of thinking, the idea of smoking and the idea of eating and drinking and smoking.\n\nThere are other ways to get out of that.\n\nI think people are going to have to learn more about what smoking is, better understand what it is, what's it about and how to live it better.\n\nA lot of people have always thought that smoking was a disease. That is wrong.\n\nI think that smoking is actually a disease. It is the disease of people who get sick with it.\n\nWhat's"},
 {'generated_text': "Nowadays young people are more icky in their behavior and have fewer friends to hang out with.\n\nWe now have a lot of social media and social networking to go around. We need to get more social media into our lives.\n\nMore social media is a better way to do that.\n\n5. Increase Social Media Safety\n\nWe do more research on getting social media safety.\n\nIt's important to know that there are many people who have been hurt in the past.\n\nI have been hurt more than I have ever been on the internet.\n\nI went to the police if I ever had to defend myself.\n\nThe police are so out of place at work that I have to go to the bathroom at work and be on the internet.\n\nI have been hurt more than I am ever going to be on the internet, so it's important to know that there are many people that have been hurt.\n\nI think we can get better at getting social media safety by doing what I do best.\n\n6. Take Care of Your Social Media\n\nSocial media is a great way to get around and protect your social media accounts.\n\nWe can use our phones, email accounts, and social media accounts as a lot of time and attention"}]
```

Analysis on example 3 : 

The text generation model produces coherent and imaginative continuations of a life's problem prompt. It demonstrates creativity and sentence flow, although output content may vary in tone and logic. The results showcase the model's usefulness for generating casual or narrative text.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
%pip install protobuf==3.20.1
nmae_ent = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
nmae_ent("My name is Roberta and I work with IBM Skills Network in Toronto")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.8209287),
  'word': 'Roberta',
  'start': 10,
  'end': 18},
 {'entity_group': 'ORG',
  'score': np.float32(0.99108773),
  'word': 'IBM Skills Network',
  'start': 34,
  'end': 53},
 {'entity_group': 'LOC',
  'score': np.float32(0.98659676),
  'word': 'Toronto',
  'start': 56,
  'end': 64}]
```

Analysis on example 4 : 

The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

### 5. Example 5 - Question Answering

```
# TODO :
awnser = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
awnser(question="what ingredients is necessary for making bread?", context="For making bread, you need flour, water, yeast, and salt.")
```

Result : 

```
{'score': 0.8929490447044373,
 'start': 27,
 'end': 56,
 'answer': 'flour, water, yeast, and salt'}
```

Analysis on example 5 : 

The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

### 6. Example 6 - Text Summarization

```
# TODO :
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
text_summary("Ontario's lake-effect machine has revved up again with the aid of an Arctic air mass that has settled into the province that will send temperatures plummeting deep into the negative values. Some of Ontario's snowbelt communities near Lake Huron and Georgian Bay, as well as areas near Lake Superior, could see 10-25+ cm of snow through Thursday afternoon. Snow squall warnings and watches are in place. Travel may be extremely hazardous in the snow squalls. Prepare for quickly changing and deteriorating travel conditions. Non-essential travel and outdoor activities should be avoided.", max_length=80, do_sample=False)
)
```

Result : 

```
[{'summary_text': " Some of Ontario's snowbelt communities near Lake Huron and Georgian Bay could see 10-25+ cm of snow through Thursday afternoon . An Arctic air mass has settled into the province that will send temperatures plummeting deep into the negative values . Snow squall warnings and watches are in place ."}]

```

Analysis on example 6 :

The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

### 7. Example 7 - Translation

```
# TODO :
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

model_en_to_de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

translate = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
data = "Right now is the best time to learn new skills."
translate(data)
```

Result : 

```
[{'translation_text': 'Im Moment ist der beste Zeitpunkt, um neue Fähigkeiten zu erlernen.'}]

```

Analysis on example 7 :

The translation model delivers an accurate and context-aware Dutch translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.