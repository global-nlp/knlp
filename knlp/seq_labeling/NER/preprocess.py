import json
from tqdm import tqdm

from knlp.common.constant import KNLP_PATH, delimiter

# -----------------------------------------------------------------------#
# File Name: __init__.py
# Author: Ziyang Miao
# Mail: 1838040569@qq.com
# Created Time: 2022-05-25
# Description: 数据预处理
# -----------------------------------------------------------------------#
from knlp.utils.tokenization import BasicTokenizer


def preprocess_trie(data_path, mid_dict_path, output_path, state_path):
    f = open(data_path, encoding='utf-8')
    out = open(mid_dict_path, 'wb', encoding='utf-8')
    flag = 0
    sign = ''
    train_data = f.readlines()
    for idx in range(len(train_data)):
        line = train_data[idx].split('\n')[0]
        nextline = ''
        if idx != len(train_data) - 1:
            nextline = train_data[idx + 1].split('\n')[0]
        if not line:
            continue
        token = line.split(delimiter)[0]
        label = line.split(delimiter)[1]

        if label != 'O':
            if nextline and nextline.split(' ')[1].split('-')[0] != 'B':
                flag = 1
                start = label.split('-')[0]
                end = label.split('-')[1]
                out.write(token.encode())
                sign = end
            else:
                start = label.split('-')[0]
                end = label.split('-')[1]
                out.write(token.encode())
                sign = end
                out.write((' ' + sign + '\n').encode())

        elif label == 'O':
            if flag == 1:
                out.write((' ' + sign + '\n').encode())
                flag = 0
            sign = ''
    f.close()
    out.close()

    f = open(mid_dict_path, encoding='utf-8')
    fo = open(output_path, 'wb', encoding='utf-8')
    repeat_dict = {}
    for line in tqdm(f.readlines()):
        token, label = line.split(' ')
        # print(token, label)
        if token not in repeat_dict.keys():
            repeat_dict[token] = 1
        else:
            repeat_dict[token] += 1
    f.close()
    f = open(mid_dict_path, encoding='utf-8')
    for line in tqdm(f.readlines()):
        token, label = line.split(' ')
        fo.write((token + ' ' + str(repeat_dict[token]) + ' ' + label).encode())
    f.close()
    fo.close()

    dict_json = json.dumps(repeat_dict, indent=2)
    f = open(state_path, "w", encoding='utf-8')
    f.write(dict_json)
    f.close()


def bio2bmes(train_path, dev_path, test_path, new_train_path, new_dev_path, new_test_path):
    train = open(train_path, 'r', encoding='utf-8')
    dev = open(dev_path, 'r', encoding='utf-8')
    test = open(test_path, 'r', encoding='utf-8')

    train_out = open(new_train_path, 'w', encoding='utf-8')
    dev_out = open(new_dev_path, 'w', encoding='utf-8')
    test_out = open(new_test_path, 'w', encoding='utf-8')

    outs = [train_out, dev_out, test_out]
    for index, i in enumerate([train, dev, test]):
        pre = i.readlines()
        for index, line in enumerate(pre):
            processed = line.replace('\n', '')
            # print(index, processed)
            if processed:
                text, label = processed.split(' ')
            else:
                outs[index].write(line)
                continue
            # print(text, label)
            if label[0] == 'I':
                if pre[index + 1] == '\n':
                    label = list(label)
                    label[0] = 'E'
                    label = ''.join(label)
                    # label.replace('I', 'E')
                    output = text + ' ' + label
                    outs[index].write(output)
                    outs[index].write('\n')
                else:
                    next_text, next_label = pre[index + 1].replace('\n', '').split(' ')
                    # print(next_text, next_label)
                    if next_label[0] == 'O' or next_label[0] == 'B':
                        label = list(label)
                        label[0] = 'E'
                        label = ''.join(label)
                        # label.replace('I', 'E')
                        output = text + ' ' + label
                        outs[index].write(output)
                        outs[index].write('\n')
                    else:
                        label = list(label)
                        label[0] = 'M'
                        label = ''.join(label)
                        output = text + ' ' + label
                        outs[index].write(output)
                        outs[index].write('\n')
            else:
                outs[index].write(processed)
                outs[index].write('\n')


def bmes2bio(train_path, dev_path, test_path, new_train_path, new_dev_path, new_test_path):
    train = open(train_path, 'r', encoding='utf-8')
    dev = open(dev_path, 'r', encoding='utf-8')
    test = open(test_path, 'r', encoding='utf-8')

    train_out = open(new_train_path, 'w', encoding='utf-8')
    dev_out = open(new_dev_path, 'w', encoding='utf-8')
    test_out = open(new_test_path, 'w', encoding='utf-8')

    outs = [train_out, dev_out, test_out]
    for index, i in enumerate([train, dev, test]):
        for line in i.readlines():
            # print(line)
            if line != '\n':
                text, label = line.strip().split('\t')
                # print(label)
                label_list = list(label)
                if label_list[0] == 'M':
                    label_list[0] = 'I'
                elif label_list[0] == 'E':
                    label_list[0] = 'I'
                new_label = ''.join(label_list)
                print(text)
                print(new_label)
                str = text + '\t' + new_label
                outs[index].write(str)
                outs[index].write('\n')
            else:
                outs[index].write(line)


class VOCABProcessor:
    """
    该类为数据集生成vocab文件，并于bert模型的vocab文件比较，添加oov词汇到vocab中。
    """
    def __init__(self, custom_datadir):
        self.wordset = set()
        self.path = custom_datadir
        self.vocab_path = self.path + 'vocab.txt'

    def gen_dict(self):
        for type in ['test', 'train', 'val']:
            f = open(self.path + f'{type}.bios', 'r', encoding='utf-8')
            for line in f.readlines():
                if line != '\n':
                    token, tag = line.strip().split(' ')
                    self.wordset.add(token)
        voc = open(self.vocab_path, 'w', encoding='utf-8')
        for item in self.wordset:
            voc.write(item + '\n')

    def merge_vocab(self):
        model = open(KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt', 'r', encoding='utf-8')
        model_text = model.readlines()
        count = []
        for index, item in enumerate(self.wordset):
            if item + '\n' not in model_text:
                count.append(item)
        print(count)

    def add_vocab(self):
        model = open(KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt', 'r', encoding='utf-8')
        voc = open(self.vocab_path, 'a+', encoding='utf-8')

        model_text = model.readlines()
        for index, item in enumerate(model_text):
            voc.write(item)


class DATAProcessor:
    """
    A unified MRC framework for named entity recognition提供的bios转mrc的脚本方法。
    """

    def __init__(self, descrip_json, vocab_path):
        with open(descrip_json, 'r', encoding='utf-8') as fp:
            labels = json.loads(fp.read())
        self.query2label = {}
        self.label2query = {}
        query = labels['default']
        for k, v in query.items():
            self.query2label[v] = k
            self.label2query[k] = v
        self.rlabel = labels['labels']
        self.basicTokenizer = BasicTokenizer(vocab_file=vocab_path, do_lower_case=True)

    def get_mid_data(self, in_path, out_path):
        with open(in_path, 'r', encoding='utf-8') as fp:
            data = fp.read()
        data = data.split('\n')
        text = ''
        entity = []
        i = 0
        start = 0
        end = 0
        tmp = ''
        res = []
        for index, d in enumerate(data):
            d = d.strip().split(delimiter)
            if index != len(data) - 1:
                next_d = data[index + 1].strip().split(delimiter)
            else:
                next_d = '\n'
            if len(d) == 2:
                text += d[0]
                if 'B-' in d[1]:
                    if len(next_d) == 1:
                        e = d[1].split('-')[-1]
                        entity.append([d[0], e, i, i])
                    else:
                        if 'B-' in next_d[1]:
                            e = d[1].split('-')[-1]
                            entity.append([d[0], e, i, i])
                        elif 'O' in next_d[1][0]:
                            e = d[1].split('-')[-1]
                            entity.append([d[0], e, i, i])
                        else:
                            start = i
                            end = i
                            tmp += d[0]
                elif 'I-' in d[1]:
                    if len(next_d) == 1:
                        e = d[1].split('-')[-1]
                        tmp += d[0]
                        end += 1
                        entity.append([tmp, e, start, end])
                        tmp = ''
                    else:
                        if 'I-' in next_d[1]:
                            tmp += d[0]
                            end += 1
                        else:
                            e = d[1].split('-')[-1]
                            tmp += d[0]
                            end += 1
                            entity.append([tmp, e, start, end])
                            tmp = ''
                i += 1
            else:
                res.append(
                    {
                        "text": text,
                        'entity': entity,
                    }
                )
                text = ''
                entity = []
                i = 0
                start = 0
                end = 0
                tmp = ''
        else:
            i += 1

        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(res, fp, ensure_ascii=False)

    def get_mrc_data(self, in_path, out_path):
        with open(in_path, 'r', encoding='utf-8') as fp:
            data = json.loads(fp.read())
        res = []
        for j, d in enumerate(data):
            text = d['text']
            entity = d['entity']
            if not entity:
                continue
            t_entity = []
            entity.sort(key=lambda x: x[2])
            for e in entity:
                left = text[:e[2]]
                t_left = self.basicTokenizer.tokenize(left)
                t_e = self.basicTokenizer.tokenize(e[0])
                t_entity.append([e[0], e[1], len(t_left), len(t_left) + len(t_e) - 1])
            for rl in self.rlabel:
                start_position = []
                end_position = []
                for i in t_entity:
                    if i[1] == rl:
                        start_position.append(i[2])
                        end_position.append(i[3])
                if j < 3:
                    print("=" * 20)
                    print("context:", " ".join(text))
                    print("entity_label:", rl)
                    print("query:", self.label2query[rl])
                    print("start_position:", start_position)
                    print("end_position:", end_position)
                    print("=" * 20)
                res.append(
                    {
                        "context": " ".join(text),
                        "entity_label": rl,
                        "query": self.label2query[rl],
                        "start_position": start_position,
                        "end_position": end_position,
                    }
                )
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(res, fp, ensure_ascii=False)


if __name__ == '__main__':
    data_path = KNLP_PATH + '/knlp/data/msra_bios/train.bios'
    mid_dict_path = KNLP_PATH + '/knlp/data/msra_bios/dict.txt'
    out_path = KNLP_PATH + '/knlp/data/msra_bios/ner_dict.txt'
    state_path = KNLP_PATH + '/knlp/data/msra_bios/state_dict.json'
    preprocess_trie(data_path, mid_dict_path, out_path, state_path)

    vocabprocessor = VOCABProcessor(KNLP_PATH + '/knlp/data/msra_bios/')
    vocabprocessor.gen_dict()
    vocabprocessor.merge_vocab()
    vocabprocessor.add_vocab()

    description_json = KNLP_PATH + '/knlp/data/msra_mrc/msra.json'
    msraProcessor = DATAProcessor(description_json)
    # 第一步：先生成中间文件和标签
    msraProcessor.get_mid_data(KNLP_PATH + '/knlp/data/msra_bios/train.bios',
                               KNLP_PATH + '/knlp/data/msra_bios/train.mid')
    msraProcessor.get_mid_data(KNLP_PATH + '/knlp/data/msra_bios/val.bios',
                               KNLP_PATH + '/knlp/data/msra_bios/dev.mid')
    msraProcessor.get_mid_data(KNLP_PATH + '/knlp/data/msra_bios/test.bios',
                               KNLP_PATH + '/knlp/data/msra_bios/test.mid')
    # 第二步：生成MRC所需要的数据
    msraProcessor.get_mrc_data(KNLP_PATH + '/knlp/data/msra_bios/train.mid',
                               KNLP_PATH + '/knlp/data/msra_mrc/train.mrc')
    msraProcessor.get_mrc_data(KNLP_PATH + '/knlp/data/msra_bios/dev.mid',
                               KNLP_PATH + '/knlp/data/msra_mrc/dev.mrc')
    msraProcessor.get_mrc_data(KNLP_PATH + '/knlp/data/msra_bios/test.mid',
                               KNLP_PATH + '/knlp/data/msra_mrc/test.mrc')
