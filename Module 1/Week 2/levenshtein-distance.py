def levenshtein(word1, word2):
    distance = [[0] * (len(word2) + 1) for i in range(len(word1) + 1)]

    for i in range(0, len(word1) + 1):
        distance[i][0] = i
    for j in range(0, len(word2) + 1):
        distance[0][j] = j

    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(distance[i - 1][j],
                                         distance[i][j - 1], distance[i - 1][j - 1])

    return distance[len(word1)][len(word2)]


if __name__ == "__main__":
    word1 = input()
    word2 = input()
    print(levenshtein(word1, word2))
