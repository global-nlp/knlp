tags = [("B-add", "I-add"), ("B-boo", "I-boo"), ("B-com", "I-com"), ("B-gam", "I-gam"), ("B-gov", "I-gov"),
        ("B-mov", "I-mov"), ("B-nam", "I-nam"), ("B-org", "I-org"), ("B-pos", "I-pos"), ("B-sce", "I-sce")]


def find_tag(labels, B_label, I_label):
    result = []
    if isinstance(labels, str):
        labels = labels.strip().split()
    for i in range(len(labels)):
        if labels[i] == B_label:
            B_pos0 = i
        if labels[i] == I_label and labels[i - 1] == B_label:
            length = 2
            for i2 in range(i, len(labels)):
                if labels[i2] == I_label and labels[i2 - 1] == I_label:
                    length += 1
                if labels[i2] == "O":
                    result.append((B_pos0, length))
                    break
    return result


def find_all_tag(labels):
    result = {}
    for tag in tags:
        res = find_tag(labels, B_label=tag[0], I_label=tag[1])
        result[tag[0].split("-")[1]] = res
    return result


def precision(pre_labels, true_labels):
    pre = []
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()

    pre_result = find_all_tag(pre_labels)
    for name in pre_result:
        for x in pre_result[name]:
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0] + x[1]]:
                    pre.append(1)
                else:
                    pre.append(0)
    return sum(pre) / len(pre)


def recall(pre_labels, true_labels):
    recall = []
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()
    true_result = find_all_tag(true_labels)
    for name in true_result:
        for x in true_result[name]:
            if x:
                if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                    recall.append(1)
                else:
                    recall.append(0)
    return sum(recall) / len(recall)


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


if __name__ == '__main__':
    fp = open('dev_pre.json', 'r')
    pre = fp.readline()
    ft = open('dev_true.json', 'r')
    true = ft.readline()
    precision = precision(pre, true)
    recall = recall(pre, true)
    f1 = f1_score(precision, recall)
    print(precision)
    print(recall)
    print(f1)
