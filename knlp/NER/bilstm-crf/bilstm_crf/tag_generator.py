import json
f = open('../../data/ner_data/cluener_training_data.txt',encoding='utf-8')
tag_set = set()
for line in f.readlines():
    if line != '\n':
        tag = line.strip().split('\t')[1]
        tag_set.add(tag)
print(tag_set)

js_dict={}
for idx,pair in enumerate(tag_set):
    js_dict[pair]=idx

with open('../bilstm_crf/model_bilstm_crf/tag_json.json', 'w') as json_file:
    json.dump(js_dict, json_file, indent=2)