class Tokenizer:
    def __init__(self, vocab, pad_token="<PAD>"):
        self.pad_token = pad_token
        self.vocab = vocab
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab)}

    def encode(self, text, max_text_length):
        encoded_sentence = [self.word_to_idx[word] for word in text]
        # If the sentence is shorter than max_text_length, pad it
        if len(encoded_sentence) < max_text_length:
            padding_needed = max_text_length - len(encoded_sentence)
            encoded_sentence += [self.word_to_idx[self.pad_token]] * padding_needed
    
        return encoded_sentence


#need to add <UNK> for words not in vocab, need to handle scenarios where text is longer than max_text_length, this class assumes text comes in as list of strings