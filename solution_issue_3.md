```python
import pandas as pd

# Пример данных
data = {
    'answer_text': [
        '参考答案是：\nA: 食物\nB: 清水',
        '参考答案是：\nA: 食物\nB: 清水',
        '参考答案是：\nA: 食物\nB: 清水',
        '参考答案是：\nA: 食物\nB: 清水',
        '参考答案是：\nA: 食物\nB: 清水'
    ],
    'answer_metadata': [
        'reference_answer: 食物\nfull_grading: [参考答案是：A: 食物\nB: 清水]',
        'reference_answer: 清水\nfull_grading: [参考答案是：清水\nB: 清水]',
        'reference_answer: 食物\nfull_grading: [参考答案是：食物\nB: 清水]',
        'reference_answer: 清水\nfull_grading: [参考答案是：清水\nB: 清水]',
        'reference_answer: 食物\nfull_grading: [参考答案是：食物\nB: 清水]'
    ]
}

df = pd.DataFrame(data)

# Очистка данных
df['answer_text'] = df['answer_text'].str.replace('\n', '')
df['answer_metadata'] = df['answer_metadata'].str.replace('\n', '')

# Отфильтровка данных
df = df[df['answer_text'] != df['answer_metadata']]

# Сохранение данных
df.to_csv('cleaned_data.csv', index=False)
```