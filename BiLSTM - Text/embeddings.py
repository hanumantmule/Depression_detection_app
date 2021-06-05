import numpy as np
import pandas as pd
import re
from allennlp.commands.elmo import ElmoEmbedder

prefix = 'E:\\Studies\\sem II\\WSC\\project\\final-app\\dataset\\transcript\\'

elmo = ElmoEmbedder()

train_split_df = pd.read_csv(prefix+'train_split_Depression_AVEC2017 (1).csv')
test_split_df = pd.read_csv(prefix+'dev_split_Depression_AVEC2017.csv')
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_clabel = train_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
test_split_clabel = test_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()

topics = []
with open('E:\\Studies\\sem II\\WSC\\project\\final-app\\dataset\\queries.txt', 'r') as f:
    for line in f.readlines():
        topics.append(line.strip('\n').strip())


def identify_topics(sentence):
    if sentence in topics:
        return True
    return True


def extract_features(number, text_features, target, mode, text_targets):
    transcript = pd.read_csv(prefix + '{0}_P/{0}_TRANSCRIPT.csv'.format(number), sep='\t').fillna('')

    responses = []
    response = ''
    response_flag = False

    global counter1, counter2

    for t in transcript.itertuples():
        if getattr(t, 'speaker') == 'Ellie':
            content = getattr(t, 'value').strip()
            if identify_topics(content):
                response_flag = True
                if len(response) != 0:
                    responses.append(response.strip())
                response = ''
            elif response_flag and len(content.split()) > 4:
                response_flag = False
                if len(response) != 0:
                    responses.append(response)
                response = ''
        elif getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                continue
            elif response_flag:
                content = getattr(t, 'value').split('\n')[0].strip()
                #                 if '<' in getattr(t,'value'):
                #                     content = re.sub(u"\\<.*?\\>", "", content)
                response += ' ' + content

    text_feature = elmo.embed_sentence(responses).mean(0)
    #     text_feature = bc.encode(responses)
    #     while text_feature.shape[0] < 30:
    #         print(number)
    #         text_feature = np.vstack((text_feature, np.zeros(text_feature.shape[1])))
    print(text_feature.shape)
    #     text_features.append(text_feature[:30])
    text_features.append(text_feature)
    text_targets.append(target)


counter1 = 0
counter2 = 0

# training set
text_features_train = []
text_ctargets_train = []

# test set
text_features_test = []
text_ctargets_test = []

# classification

#training set
for index in range(len(train_split_num)):
    extract_features(train_split_num[index], text_features_train, train_split_clabel[index], 'train',
                     text_ctargets_train)

# test set
for index in range(len(test_split_num)):
    extract_features(test_split_num[index], text_features_test, test_split_clabel[index], 'test', text_ctargets_test)


#saving files locally

text_features_train = np.load(prefix+'data\\train_samples.npz')['arr_0']
text_features_test = np.load(prefix+'data\\train_labels.npz')['arr_0']
text_ctargets_train = np.load(prefix+'data\\test_samples.npz')['arr_0']
text_ctargets_test = np.load(prefix+'data\\test_labels.npz')['arr_0']