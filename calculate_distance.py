import numpy as np
import scipy.spatial.distance as dist

# 计算编辑距离


def calculateLevenshteinDistance(word1, word2):
    matrix = [[i + j for j in range(len(word2) + 1)]
              for i in range(len(word1) + 1)]
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i]
                               [j - 1] + 1, matrix[i - 1][j - 1] + d)
    print('The Levenshtein Distance between ' + word1 + ' and ' + word2 +
          ' is:', 1 - matrix[len(word1)][len(word2)] / max(len(word1), len(word2)))

    # if len(word1)>=len(word2):
    #     return 1-matrix[len(word1)][len(word2)]/len(word1)
    # else:
    #     return 1-matrix[len(word1)][len(word2)]/len(word1)
    # xlen = len(word1)+1
    # ylen = len(word2)+1
    # dp = np.zeros(shape=(xlen, ylen), dtype=int)
    # for i in range(0, xlen):
    #     dp[i][0] = i
    # for j in range(0, ylen):
    #     dp[0][j] = j
    # for i in range(1, xlen):
    #     for j in range(1, ylen):
    #         if word1[i - 1] == word2[j - 1]:
    #             dp[i][j] = dp[i - 1][j - 1]
    #         else:
    #             dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    #

# 计算汉明距离


def calculateHammingDistance(vec1, vec2):
    smstr = np.nonzero(vec1 - vec2)
    print(smstr)  # 不为0 的元素的下标
    sm = np.shape(smstr[0])[0]
    print('汉明距离：', sm)


# 计算杰卡德相似系数(Jaccard similarity coefficient)
def calculateJaccard(vec1, vec2):
    matv = np.array([vec1, vec2])
    print(matv)
    ds = dist.pdist(matv, 'jaccard')
    print(ds)

# 计算欧式距离（Euclidean Distance）

# 计算欧式距离
def calculateEuclideanDistance(vec1, vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    # 直接调用函数
    # op2 = np.linalg.norm(vec1 - vec2)
    print('EuclideanDistance:', dist)

# 计算余弦相似度
def calculateCosineDistance(vec1, vec2):
    dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * (np.linalg.norm(vec2)))
    print('Cosine:', dist)

# 计算切比雪夫距离
def calculateChebyshevDistance(vec1, vec2):
    dist = np.abs(vec1 - vec2).max()
    dist = np.linalg.norm(vec1 - vec2, ord=np.inf)
    print('ChebyshevDistance:', dist)

# 接收离散型变量的向量


calculateLevenshteinDistance('good', 'yes')
calculateLevenshteinDistance('large', 'later')
calculateLevenshteinDistance('web', 'web')
calculateLevenshteinDistance('web-science', 'nlp')
# exit()
# 接收连续型变量的向量
wordEmbedding_dict = {}
for line in open('model/word2vec.txt', 'r', encoding='utf-8'):
    templist = line.strip().split(' ')
    wordEmbedding_dict[templist[0]] = list(map(float, templist[1:]))
vec1 = np.array(wordEmbedding_dict['july'])
for item in wordEmbedding_dict:
    print('july  &  ', item)
    vec2 = np.array(wordEmbedding_dict[item])
    calculateEuclideanDistance(vec1, vec2)
    calculateCosineDistance(vec1, vec2)
    calculateChebyshevDistance(vec1, vec2)
    print('\n')
