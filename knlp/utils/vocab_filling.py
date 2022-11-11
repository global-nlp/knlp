from knlp.common.constant import KNLP_PATH

dev = open(KNLP_PATH + '/knlp/data/bios_clue/train.bios', 'r', encoding='utf-8')
out = open(KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt', 'r', encoding='utf-8')

# print(dev.readlines())
# print(out.readlines())
devv = []
for line in dev.readlines():
    if line[0] != '\n': devv.append(line[0])

outt = []
for line in out.readlines():
    outt.append(line[:-1])

set1 = set(devv)
set2 = set(outt)

u = set1.difference(set2)

print(u)
dev.close()
out.close()

f = open(KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt', 'r', encoding='utf-8')
list_of_lines = f.readlines()
for idx, line in enumerate(list_of_lines):
    if len(u) == 0:
        break
    if line.startswith('[u'):
        list_of_lines[idx] = u.pop() + '\n'

f = open(KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt', 'w', encoding='utf-8')
f.writelines(list_of_lines)