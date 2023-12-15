import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")
import numpy as np

class OneHotEncoder:
    def __init__(self, padder=' ', max_length=None):
        self.padder = padder
        self.max_length = max_length
        self.alphabet = set()
        self.char_to_index = {}
        self.index_to_char = {}

    def fit(self, data):
        self.alphabet = sorted(set("".join(data)))
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for char, index in self.char_to_index.items()}
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)

    def transform(self, data):
        encoded_sequences = []
        for seq in data:
            seq = seq[:self.max_length].ljust(self.max_length, self.padder)
            encoded_seq = np.zeros((self.max_length, len(self.alphabet)))
            for i, char in enumerate(seq):
                if char in self.char_to_index:
                    encoded_seq[i][self.char_to_index[char]] = 1
            encoded_sequences.append(encoded_seq)
        return np.array(encoded_sequences)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        decoded_sequences = []
        for seq in data:
            decoded_seq = ''
            for char_encoding in seq:
                char_index = np.argmax(char_encoding)
                if char_index in self.index_to_char:
                    decoded_seq += self.index_to_char[char_index]
            decoded_sequences.append(decoded_seq)
        return decoded_sequences
    

if __name__ == '__main__':
    # Example usage
    data = ['cat', 'dog', 'elephant']
    encoder = OneHotEncoder(max_length=10)
    
    # Fit and transform the data
    encoded_data = encoder.fit_transform(data)
    print("Encoded Data:")
    print(encoded_data)
    
    # Inverse transform the encoded data
    decoded_data = encoder.inverse_transform(encoded_data)
    print("\nDecoded Data:")
    print(decoded_data)