from keras.preprocessing.text import Tokenizer

def vocabulary_count(text: list[str]):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return len(tokenizer.word_counts)



