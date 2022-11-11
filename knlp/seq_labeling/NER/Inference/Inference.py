class Inference:
    def __init__(self):
        self.out_sent = None
        self.tag_list = []
        self.entity_set = set()
        self.log = False

    def get_tag(self):
        return self.tag_list

    def get_entity(self):
        return self.entity_set

    def get_sent(self):
        return self.out_sent

    def detailed_log(self):
        self.log = True

    def cut_bio(self, sentence1, sentence2):
        """
        按照BIO标签做中文分词，切割句子。
        Args:
            sentence1: 文本序列
            sentence2: 标注序列

        Returns:

        """
        out_sent = []
        begin = 0
        self.tag_list.append(sentence2)
        for idx in range(len(sentence1)):
            if sentence2[idx][0] == 'B' and sentence2[idx+1][0] == 'O':
                out_sent.append(sentence1[idx])
                self.entity_set.add((sentence1[idx], sentence2[idx][2:]))
                continue
            if sentence2[idx][0] == 'B':
                begin = idx
            elif sentence2[idx][0] == 'I':
                idx += 1
                if idx == len(sentence1):
                    str = "".join(sentence1[begin:idx])
                    out_sent.append(str)
                    self.entity_set.add((str, sentence2[idx-1][2:]))
                    self.out_sent = out_sent
                    # print(out_sent)
                elif sentence2[idx][0] == 'O' or sentence2[idx][0] == 'B':
                    str = "".join(sentence1[begin:idx])
                    out_sent.append(str)
                    self.entity_set.add((str, sentence2[idx-1][2:]))
                    begin = idx
            elif sentence2[idx][0] == 'O':
                out_sent.append(sentence1[idx])
            # print(out_sent)
        self.out_sent = out_sent

    def cut_bmes(self, sentence1, sentence2):
        """
        按照BIO标签做中文分词，切割句子。
        Args:
            sentence1: 文本序列
            sentence2: 标注序列

        Returns:

        """
        out_sent = []
        begin = 0
        self.tag_list.append(sentence2)
        for idx in range(len(sentence1)):
            #    print(sentence1[idx], sentence2[idx])
            if sentence2[idx][0] == 'B':
                begin = idx
            elif sentence2[idx][0] == 'M':
                idx += 1
                if idx == len(sentence1):
                    str = "".join(sentence1[begin:idx])
                    out_sent.append(str)
                    # print(out_sent)
                    self.out_sent = out_sent
                elif sentence2[idx][0] == 'E':
                    str = "".join(sentence1[begin:idx + 1])
                    out_sent.append(str)
                    self.entity_set.add((str, sentence2[idx][0:-1]))
                    begin = idx + 1
            elif sentence2[idx][0] == 'E':
                str = "".join(sentence1[begin:idx + 1])
                out_sent.append(str)
                self.entity_set.add((str, sentence2[idx][0:-1]))
                begin = idx + 1
            elif sentence2[idx][0] == 'O':
                out_sent.append(sentence1[idx])
        # print(out_sent)
        self.out_sent = out_sent