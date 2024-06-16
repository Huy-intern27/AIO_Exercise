def max_kernel(num_list, k):
    result = []
    slide_list = []

    for i in range(k):
        slide_list.append(num_list[i])

    max_element = max(slide_list)
    result.append(max_element)

    for i in range(k, len(num_list)):
        slide_list.pop(0)
        slide_list.append(num_list[i])
        if max_element < num_list[i]:
            max_element = num_list[i]
        result.append(max_element)

    return result

if __name__ == "__main__":
    num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
    k = 3
    print(max_kernel(num_list, k))