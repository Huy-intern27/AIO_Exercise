def count_chars(string_count):
    count_words = {}
    for element in string_count:
        if element not in count_words:
            count_words[element] = string_count.count(element)

    print(count_words)


if __name__ == "__main__":
    string = input()
    count_chars(string)
