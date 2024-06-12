if __name__ == "__main":
    num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
    k = 3

    result = []
    slide_list = []
    max_ele = -1e9

    for i in range(k):
        slide_list.append(num_list[i])

    max_ele = max(slide_list)
    result.append(max_ele)

    for i in range(k, len(num_list)):
        slide_list.pop(0)
        slide_list.append(num_list[i])
        if max_ele < num_list[i]:
            max_ele = num_list[i]
        result.append(max_ele)

    for element in result:
        print(element, end=' ')
