from torch.utils.data import Dataset
import torch
import pandas as pd

from pathlib import Path
import os

import numpy as np

characters = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 
15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36, '-': 61, ',': 38, ';': 39, '.': 40, '!': 41, '?': 42, ':': 43, '’': 46, '/': 47, '\\': 48, '|': 49, '_': 50, '@': 51, '#': 52, '$': 53, '%': 54, 'ˆ': 55, '&': 56, '*': 57, '˜': 58, '‘': 59, '+': 60, '=': 62, '<': 63, '>': 64, '(': 65, ')': 66, '[': 67, ']': 68, '{': 69, '}': 70}

len_dictionary = 70

path = Path(__file__).parent.parent

def word_to_one_hot(string, max_embedding):
    torch_string = None
    for index, char in enumerate(string):
        if index >= max_embedding:
            break

        torch_tmp = torch.zeros((1, len_dictionary))
        if char in characters.keys():
            torch_tmp[:, characters[char]] = 1

        if index == 0:
            torch_string = torch_tmp
        else:
            torch_string = torch.cat((torch_string, torch_tmp), dim=0)
    
    if len(string) < max_embedding:
        blank_torch = torch.zeros((max_embedding - len(string), len_dictionary))
        torch_string = torch.cat((torch_string, blank_torch), dim=0)

    return torch_string
            
class WordDataset(Dataset):
    def __init__(self, type_ds="train", max_embed_length=10, words_next_to=1):
        if type_ds in ["train", "testa", "testb"]:
            self.annotate = pd.read_csv(os.path.join(path, "data", "eng_{}.csv".format(type_ds)))
        else:
            raise Exception("Unallowed type of dataset")
        
        self.max_embed_length = max_embed_length
        self.words_next_to = words_next_to

    def __len__(self):
        return len(self.annotate)
    
    def padding(self, strings, pad_before, pad_after):
        total_torch = None

        for index, string in enumerate(strings):
            torch_string = torch.zeros((self.max_embed_length, len_dictionary))
            if not isinstance(string, float):
                torch_string = word_to_one_hot(string, max_embedding=self.max_embed_length)

            if index == 0:
                total_torch = torch_string
            else:
                blank_torch = torch.zeros((1, len_dictionary))
                total_torch = torch.cat((total_torch, blank_torch, torch_string), dim = 0)

        #if type(total_torch) == type(None):
            #print(strings, pad_before, pad_after)
        
        if pad_before > 0:
            before_torch = torch.zeros(((self.max_embed_length + 1) * pad_before, len_dictionary))
            total_torch = torch.cat((before_torch, total_torch), dim=0)

        if pad_after > 0:
            after_torch = torch.zeros(((self.max_embed_length + 1) * pad_after, len_dictionary))
            total_torch = torch.cat((total_torch, after_torch), dim=0)

        return total_torch
        

    def __getitem__(self, index):
        row = self.annotate.iloc[index]
        first_word_of_sentence_id = row["first_word_of_sentence_id"]
        #print(np.arange(max(index - self.words_next_to, first_word_of_sentence_id), index + self.words_next_to + 1))
        #print(index - self.words_next_to, first_word_of_sentence_id, index + self.words_next_to + 1)
        range_to_take = np.arange(max(index - self.words_next_to, first_word_of_sentence_id), min(self.__len__(), index + self.words_next_to + 1))
        strings = self.annotate.iloc[range_to_take]
        #print(strings["word"])
        strings = strings.loc[self.annotate["first_word_of_sentence_id"] == first_word_of_sentence_id]["word"]
        #print(strings)
        #print("----------------------------------")
        label = torch.tensor(row["type"], dtype=torch.int64)

        pad_before = 0
        pad_after = 0
        total_len = len(strings)

        if index - first_word_of_sentence_id < self.words_next_to:
            total_len += (self.words_next_to - index + first_word_of_sentence_id)
            pad_before = (self.words_next_to - index + first_word_of_sentence_id)

        if total_len < self.words_next_to * 2 + 1:
            pad_after = self.words_next_to * 2 + 1 - total_len

        data = self.padding(strings, pad_before, pad_after)

        return (data, label)
    
if __name__ == "__main__":
    a = WordDataset()
    data, label = a.__getitem__(2)
    print(data.shape)
    data, label = a.__getitem__(113)
    print(data.shape)
    data, label = a.__getitem__(214)
    print(data.shape)