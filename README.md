# keyword-extraction-project

## Структура проекта
```
.
├── Dockerfile
├── README.md
├── commands.py
├── keywords_extraction
│   ├── infer.py
│   └── train.py
├── poetry.lock
└── pyproject.toml

```

## Описание задачи и данных

Решается задачи выделения ключевых слов (тэгов). Почему это может быть полезно? Для решения задачи был выбран датасет с вопросами из StackOverflow, и конкретно для StackOverflow может быть полезно уметь автоматически проставлять тэги в случае, когда люди этого не сделали или не до конца их проставили. В общем же случае, выделение ключевых слов - довольно важная задача, если мы хотим делать какой-то Topic Modelling.

Задача сама по себе довольно обширня, поэтому решила сосредоточиться на выделению ключевых слов из вопросов.


### Описаниe датасета

Датасет состоит из вопросов из StackOverflow, для которых нужно определить тэги. Ссылка на датасет: https://www.kaggle.com/competitions/facebook-recruiting-iii-keyword-extraction/data

Id - идентификатор вопроса

Title - название вопроса

Body - тело вопроса

Tags - тэги, ассоциированные с этим вопросом

Особенность: датасет будет содержать куски кода и это может быть проблемой, так как в самом коде нет обычно ключевых слов, которые могут быть тэгам, а этот код может занимать бОльшую часть вопроса.
Однако скорее всего, таких вопросов будет не очень много. 

Почему этого будет достаточно: StackOVerflow - одна из самых больших платформ (из аналогов есть Quora), где человек может задать вопрос, поэтому датасета вопросов на этой платформе должно быть достаточно, чтобы решить задачу.

## Описание решения

Так как предлагалось использовать только решения с нейросетями, то данную задачу планирую решать с помощью дообучения BERT-like модели на данных из датасета, скорее всего это будет longformer или xlm-roberta. Возможно еще для сравнения попробую BiLSTM и выберу из этих двух подходов тот, на котором будет более качественная классификация.

