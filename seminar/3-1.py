from gensim.models import Word2Vec

# 훈련에 사용된 데이터
training_data = [
    ['강아지', '고양이', '두', '마리', '계단', '위', '앉아', '있다']
]

# Word2Vec 사용하여 벡터로 변환
word2vec_model = Word2Vec(sentences = training_data, min_count = 1)

word_vector = word2vec_model.wv['강아지'] # 강아지를 벡터로 변환

print(word_vector)