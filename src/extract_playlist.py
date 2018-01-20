from utils import *
import gensim

def extract_playlist(df_30):
    sentence_list = []
    temp_sentence = []
    idx = 0

    while idx < len(df_30):
        current_row = df_30.iloc[idx]
        next_row = df_30.iloc[idx+1]
        username = current_row.uid
        while next_row.uid == username and int(next_row.timestamp) < int(current_row.timestamp) + int(current_row.playtime) + 300:
            if int(current_row.playtime) > 9:
                temp_sentence.append(current_row.tid)
            idx = idx+1
            current_row = df_30.iloc[idx]
            next_row = df_30.iloc[idx+1]
            if idx % 50000 == 0:
                print(idx)
        temp_sentence.append(current_row.tid)
        idx = idx + 1
        sentence_list.append(temp_sentence)
        temp_sentence = []

    sentence_list = [x for x in sentence_list if len(x) > 1]
    save_pickle(sentence_list, 'sentence_list_30')
    return sentence_list


def train_song2vec(sentences, min_ct):
    model = gensim.models.Word2Vec(sentences, min_count=min_ct)

    return model

if __name__ == "__main__":
    df_30 = check_and_import('./df/df_30_raw')
    if df_30 is None:
        df_30 = import_30("./data/30_eventinfo.csv")
        save_pickle(df_30, 'df_30_raw')

    sentences = extract_playlist(df_30)
    song2vec = train_song2vec(sentences, 5)
    
    # song2vec['1']
    # song2vec.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    # song2vec.doesnt_match("breakfast cereal dinner lunch";.split())
    # song2vec.similarity('woman', 'man')
    
