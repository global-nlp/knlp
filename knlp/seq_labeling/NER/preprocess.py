import json
from tqdm import tqdm

from knlp.common.constant import KNLP_PATH, delimiter

# -----------------------------------------------------------------------#
# File Name: __init__.py
# Author: Ziyang Miao
# Mail: 1838040569@qq.com
# Created Time: 2022-05-25
# Description: 用于trie数据预处理
# -----------------------------------------------------------------------#
from knlp.utils.tokenization import BasicTokenizer

basicTokenizer = BasicTokenizer(vocab_file=KNLP_PATH + '/knlp/data/msra_bios/vocab.txt', do_lower_case=True)


def preprocess_trie(data_path, output_path, state_path):
    f = open(data_path, encoding='utf-8')
    out = open(KNLP_PATH + '/knlp/data/NER_data/dict.txt', 'wb')
    flag = 0
    sign = ''
    train_data = f.readlines()
    for idx in range(len(train_data)):
        # print(line)
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

    f = open(KNLP_PATH + '/knlp/data/NER_data/dict.txt', encoding='utf-8')
    fo = open(output_path, 'wb')
    repeat_dict = {}
    for line in tqdm(f.readlines()):
        token, label = line.split(' ')
        # print(token, label)
        if token not in repeat_dict.keys():
            repeat_dict[token] = 1
        else:
            repeat_dict[token] += 1
    f.close()
    f = open(KNLP_PATH + '/knlp/data/NER_data/dict.txt', encoding='utf-8')
    for line in tqdm(f.readlines()):
        token, label = line.split(' ')
        fo.write((token + ' ' + str(repeat_dict[token]) + ' ' + label).encode())
    f.close()
    fo.close()

    dict_json = json.dumps(repeat_dict, indent=2)
    f = open(state_path, "w")
    f.write(dict_json)
    f.close()


class DATAProcessor:
    def __init__(self, descrip_json):
        with open(descrip_json, 'r') as fp:
            labels = json.loads(fp.read())
        self.query2label = {}
        self.label2query = {}
        query = labels['default']
        for k, v in query.items():
            self.query2label[v] = k
            self.label2query[k] = v
        self.rlabel = labels['labels']

    def get_mid_data(self, in_path, out_path):
        with open(in_path, 'r') as fp:
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
            if index != len(data)-1:
                next_d = data[index+1].strip().split(delimiter)
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

        with open(out_path, 'w') as fp:
            json.dump(res, fp, ensure_ascii=False)

    def get_mrc_data(self, in_path, out_path):
        with open(in_path, 'r') as fp:
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
                t_left = basicTokenizer.tokenize(left)
                t_e = basicTokenizer.tokenize(e[0])
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
        with open(out_path, 'w') as fp:
            json.dump(res, fp, ensure_ascii=False)


if __name__ == '__main__':
    data_path = KNLP_PATH + '/knlp/data/msra_bios/train.bios'
    out_path = KNLP_PATH + '/knlp/data/NER_data/ner_dict.txt'
    state_path = KNLP_PATH + '/knlp/data/NER_data/state_dict.json'
    preprocess_trie(data_path, out_path, state_path)

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
