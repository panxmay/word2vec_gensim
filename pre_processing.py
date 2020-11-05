import re
import nltk
nltk.download('punkt')


def read_txt_to_list(path, code):
    data_list = []
    with open(path, 'r', encoding=code) as f:
        for line in f:
            data_list.append(line.strip())
    return data_list


def read_txt_to_dict(path, code, gap):
    data_dict = {}
    with open(path, 'r', encoding=code) as f:
        for line in f:
            line_list = line.split(gap)
            if len(line_list) < 2:
                continue
            ca_ab_list = line_list[1].replace(
                '[', '').replace(
                ']', '').replace(
                ',', '').split(',')
            data_dict[line_list[0]] = ca_ab_list
    return data_dict


def write_to_txt(data, path):
    f = open(path, 'w', encoding='utf-8')
    for line in data:
        f.write(line)
    f.close()


def write_dict_to_txt(data, path, gap):
    f = open(path, 'w+', encoding='utf-8')
    for line in data:
        string = str(line) + gap + str(data[line]) + '\n'
        f.write(string)
    f.close()


# 预处理要完成的任务：单词标记化、小写、提取原始文本、分词、去掉停用词

def split_words(text):
    sents = nltk.sent_tokenize(text)
    words = []
    for sent in sents:
        words.append(nltk.word_tokenize(sent))
    return words


def pre_processing():
    word_description_dict = {}
    source_path = 'sample_data/wiki_00'
    with open(source_path, 'r', encoding='utf-8') as f:
        pattern = re.compile(r'>\s+([^<]+)\s+<')
        str_list = re.findall(pattern, str(f.read()))
    for item in str_list:
        temp_list = str(item).split('\n\n')
        word_describe = ''
        for i in range(1, len(temp_list)):
            word_describe += str(temp_list[i])
        word_description_dict[temp_list[0].lower()] = split_words(
            word_describe.lower())
    return word_description_dict


content = pre_processing()
write_dict_to_txt(content, 'sample_data/processed_data.txt', ' ')
