import pandas as pd

classes_type = {}

def to_csv_from_txt(file_to_open, file_destination):
    with open(file_to_open, "r") as file:
        dict_data = []
        first_word_of_sentence_id = 0
        current_sample = 0
        for index, line in enumerate(file.readlines()):
            try:    
                word, type_word = line.split(" ")

                if not type_word[:-1] in classes_type.keys():
                    if index < 5:
                        print(type_word[:-1], classes_type)
                    classes_type[type_word[:-1]] = len(classes_type.keys()) + 1
                    if index < 5:
                        print(type_word[:-1], classes_type)

                dict_data.append({
                    "word": word.lower(),
                    "type": classes_type[type_word[:-1]],
                    "first_word_of_sentence_id": first_word_of_sentence_id
                })
                
                if word == ".":
                    first_word_of_sentence_id = current_sample + 1

                current_sample += 1
            except Exception as e:
                
                if index < 5:
                    print(e)
                pass

        
        df = pd.DataFrame.from_dict(dict_data)
        print(df.head())
        print(len(df))
        print(len(df["first_word_of_sentence_id"].unique()))
        df.to_csv(file_destination, index=False)

to_csv_from_txt("./eng.train", "./eng_train.csv")
to_csv_from_txt("./eng.testa", "./eng_testa.csv")
to_csv_from_txt("./eng.testb", "./eng_testb.csv")