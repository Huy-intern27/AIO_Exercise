def word_count(path):
    with open(path, 'r') as file:
        data = file.read().replace('\n', ' ').lower()
        list_word = data.split()
        result = {}
        for token in list_word:
            if token not in result:
                result[token] = list_word.count(token)
        print(result)


if __name__ == "__main__":
    file_path = 'Module 1/Week 2/Work with file/data.txt'
    word_count(file_path)
