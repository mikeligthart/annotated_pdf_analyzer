# import gensim
# from gensim import corpora
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import re
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer


# def preprocess_text_LDA(sentences):
#     stop_words = set(stopwords.words('english'))
#     # Remove punctuation and tokenize sentences
#     sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
#     # Tokenize the sentences and remove stopwords
#     tokenized_sentences = [sentence.lower().split() for sentence in sentences]
#     filtered_sentences = [[word for word in tokens if word not in stop_words] for tokens in tokenized_sentences]
#     return filtered_sentences

# def train_lda_model(corpus, num_topics=3, passes=30, random_state=42):
#     # Create a dictionary from the tokenized sentences
#     dictionary = corpora.Dictionary(corpus)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in corpus]
#     # Train the LDA model
#     lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                                  id2word=dictionary,
#                                                  num_topics=num_topics,
#                                                  random_state=random_state,
#                                                  passes=passes)
#     return lda_model, dictionary

# def analyze_topics(sentences, lda_model, dictionary):
#     doc_topics_all = []
#     for sentence in sentences:
#         bow = dictionary.doc2bow(sentence)
#         doc_topics = lda_model.get_document_topics(bow)
#         doc_topics_all.extend(doc_topics)
#     return doc_topics_all


# # Preprocess text
# def preprocess_text_LSA(text_list):
#     preprocessed_texts = []
#     for text in text_list:
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#         tokens = text.lower().split()  # Tokenize
#         tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
#         preprocessed_texts.append(' '.join(tokens))
#     return preprocessed_texts



# if __name__ == '__main__':
#     stop_words = stopwords.words('english')
#     stop_words.extend(['also', 'like', 'could'])
#     df = pd.read_csv("output/scenario_dataframe.csv")
#         # Preprocess scenario highlights and comments separately
#     scenario_highlights = {
#         'Bert has to stay': df[df['title_scenario'] == 'Bert has to stay']['annotation_sentence'].values,
#         'Marco will be fine': df[df['title_scenario'] == 'Marco Will Be Fine']['annotation_sentence'].values,
#         'Morning aerobics': df[df['title_scenario'] == 'Morning Aerobics']['annotation_sentence'].values,
#         'Winter': df[df['title_scenario'] == 'Winter']['annotation_sentence'].values
#     }

#     scenario_comments = {
#         'Bert has to stay': df[df['title_scenario'] == 'Bert has to stay']['annotation_comment'].values,
#         'Marco will be fine': df[df['title_scenario'] == 'Marco Will Be Fine']['annotation_comment'].values,
#         'Morning aerobics': df[df['title_scenario'] == 'Morning Aerobics']['annotation_comment'].values,
#         'Winter': df[df['title_scenario'] == 'Winter']['annotation_comment'].values
#     }


#     print("Separate scenario-wise analysis:")

#     #Method 1: LDA Latent Dirichlet allocation
#     print("LDA METHOD:")
#     for scenario, highlights in scenario_highlights.items():
#         print(f"Scenario: {scenario}")
#         preprocessed_highlights = preprocess_text_LDA(highlights)
#         lda_model_highlights, dictionary_highlights = train_lda_model(preprocessed_highlights)
#         print("Topics based on highlights:")
#         scenario_topics_highlights = analyze_topics(preprocessed_highlights, lda_model_highlights, dictionary_highlights)
#         topic_distribution_highlights = {}
#         for topic_num, prob in scenario_topics_highlights:
#             if topic_num not in topic_distribution_highlights:
#                 topic_distribution_highlights[topic_num] = prob
#             else:
#                 topic_distribution_highlights[topic_num] += prob
#         total_documents_highlights = len(preprocessed_highlights)
#         for topic_num, prob_sum in topic_distribution_highlights.items():
#             prob_avg = prob_sum / total_documents_highlights
#             print(f"  Topic {topic_num}: Average Probability {prob_avg:.4f}")
#             print("    Top words:", ", ".join([word for word, _ in lda_model_highlights.show_topic(topic_num)]))
#         print()
        
#         # Perform LDA on comments
#         comments = scenario_comments[scenario]
#         preprocessed_comments = preprocess_text_LDA(comments)
#         lda_model_comments, dictionary_comments = train_lda_model(preprocessed_comments)
#         print("Topics based on comments:")
#         scenario_topics_comments = analyze_topics(preprocessed_comments, lda_model_comments, dictionary_comments)
#         topic_distribution_comments = {}
#         for topic_num, prob in scenario_topics_comments:
#             if topic_num not in topic_distribution_comments:
#                 topic_distribution_comments[topic_num] = prob
#             else:
#                 topic_distribution_comments[topic_num] += prob
#         total_documents_comments = len(preprocessed_comments)
#         for topic_num, prob_sum in topic_distribution_comments.items():
#             prob_avg = prob_sum / total_documents_comments
#             print(f"  Topic {topic_num}: Average Probability {prob_avg:.4f}")
#             print("    Top words:", ", ".join([word for word, _ in lda_model_comments.show_topic(topic_num)]))
#         print()
#         print()
#     print()
        


#     #Method 2: LSA Latent semantic analysis
#     # Preprocess scenario highlights and comments separately
#     preprocessed_highlights = {scenario: preprocess_text_LSA(highlights) for scenario, highlights in scenario_highlights.items()}
#     preprocessed_comments = {scenario: preprocess_text_LSA(comments) for scenario, comments in scenario_comments.items()}

#     # Vectorize text data using TF-IDF separately for highlights and comments
#     vectorizer_highlights = TfidfVectorizer()
#     vectorizer_comments = TfidfVectorizer()

#     # Train separate LSA models for highlights and comments
#     num_components = 3  # Number of components to retain
#     lsa_model_highlights = TruncatedSVD(n_components=num_components, random_state=42)
#     lsa_model_comments = TruncatedSVD(n_components=num_components, random_state=42)


#     print("LSA METHOD:")
#     # Get top words or themes for scenario highlights and comments separately
#     for scenario, highlights in preprocessed_highlights.items():
#         tfidf_matrix_highlights = vectorizer_highlights.fit_transform(highlights)
#         lsa_matrix_highlights = lsa_model_highlights.fit_transform(tfidf_matrix_highlights)

#         print(f"Scenario:  {scenario}")
        
#         print("Top words for scenario highlights:")
#         feature_names_highlights = vectorizer_highlights.get_feature_names_out()
#         for i in range(3):  # Print three themes per scenario highlights
#             print(f"Theme {i+1}: {', '.join(feature_names_highlights[idx] for idx in lsa_model_highlights.components_[i].argsort()[-8:][::-1])}")
#         print()
        
#         comments = preprocessed_comments[scenario]
#         tfidf_matrix_comments = vectorizer_comments.fit_transform(comments)
#         lsa_matrix_comments = lsa_model_comments.fit_transform(tfidf_matrix_comments)
        
#         print("Top words for scenario comments:")
#         feature_names_comments = vectorizer_comments.get_feature_names_out()
#         for i in range(3):  # Print three themes per scenario comments
#             print(f"Theme {i+1}: {', '.join(feature_names_comments[idx] for idx in lsa_model_comments.components_[i].argsort()[-8:][::-1])}")
#         print()
#         print()


from gensim import corpora
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim


def preprocess_text_LDA(sentences):
    stop_words = set(stopwords.words('english'))
    sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    filtered_sentences = [[word for word in tokens if word not in stop_words] for tokens in tokenized_sentences]
    return filtered_sentences


def preprocess_text_LSA(text_list):
    preprocessed_texts = []
    for text in text_list:
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.lower().split()
        tokens = [word for word in tokens if word not in stop_words]
        preprocessed_texts.append(' '.join(tokens))
    return preprocessed_texts


def train_lda_model(corpus, num_topics=3, passes=30, random_state=42):
    dictionary = corpora.Dictionary(corpus)
    corpus = [dictionary.doc2bow(tokens) for tokens in corpus]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 random_state=random_state,
                                                 passes=passes)
    return lda_model, dictionary


def analyze_topics(sentences, lda_model, dictionary):
    doc_topics_all = []
    for sentence in sentences:
        bow = dictionary.doc2bow(sentence)
        doc_topics = lda_model.get_document_topics(bow)
        doc_topics_all.extend(doc_topics)
    return doc_topics_all


if __name__ == '__main__':
    stop_words = stopwords.words('english')
    stop_words.extend(['also', 'like', 'could'])
    df = pd.read_csv("output/scenario_dataframe.csv")

    scenario_highlights = {
        'Bert has to stay': df[df['title_scenario'] == 'Bert has to stay']['annotation_sentence'].values,
        'Marco will be fine': df[df['title_scenario'] == 'Marco Will Be Fine']['annotation_sentence'].values,
        'Morning aerobics': df[df['title_scenario'] == 'Morning Aerobics']['annotation_sentence'].values,
        'Winter': df[df['title_scenario'] == 'Winter']['annotation_sentence'].values,
        'Combined scenarios': df['annotation_sentence'].values
    }

    scenario_comments = {
        'Bert has to stay': df[df['title_scenario'] == 'Bert has to stay']['annotation_comment'].values,
        'Marco will be fine': df[df['title_scenario'] == 'Marco Will Be Fine']['annotation_comment'].values,
        'Morning aerobics': df[df['title_scenario'] == 'Morning Aerobics']['annotation_comment'].values,
        'Winter': df[df['title_scenario'] == 'Winter']['annotation_comment'].values,
        'Combined scenarios': df['annotation_comment'].values

    }

    print(df[df['title_scenario'] == 'Bert has to stay']['annotation_sentence'].values)

    print("Separate scenario-wise analysis:")

    # Method 1: LDA Latent Dirichlet Allocation
    print("LDA METHOD:")
    for scenario, highlights in scenario_highlights.items():
        print(f"Scenario: {scenario}")
        preprocessed_highlights = preprocess_text_LDA(highlights)
        lda_model_highlights, dictionary_highlights = train_lda_model(preprocessed_highlights)
        print("Topics based on highlights:")
        scenario_topics_highlights = analyze_topics(preprocessed_highlights, lda_model_highlights, dictionary_highlights)
        topic_distribution_highlights = {}
        for topic_num, prob in scenario_topics_highlights:
            if topic_num not in topic_distribution_highlights:
                topic_distribution_highlights[topic_num] = prob
            else:
                topic_distribution_highlights[topic_num] += prob
        total_documents_highlights = len(preprocessed_highlights)
        for topic_num, prob_sum in topic_distribution_highlights.items():
            prob_avg = prob_sum / total_documents_highlights
            print(f"  Topic {topic_num}: Average Probability {prob_avg:.4f}")
            print("    Top words:", ", ".join([word for word, _ in lda_model_highlights.show_topic(topic_num)]))
        print()

        comments = scenario_comments[scenario]
        preprocessed_comments = preprocess_text_LDA(comments)
        lda_model_comments, dictionary_comments = train_lda_model(preprocessed_comments)
        print("Topics based on comments:")
        scenario_topics_comments = analyze_topics(preprocessed_comments, lda_model_comments, dictionary_comments)
        topic_distribution_comments = {}
        for topic_num, prob in scenario_topics_comments:
            if topic_num not in topic_distribution_comments:
                topic_distribution_comments[topic_num] = prob
            else:
                topic_distribution_comments[topic_num] += prob
        total_documents_comments = len(preprocessed_comments)
        for topic_num, prob_sum in topic_distribution_comments.items():
            prob_avg = prob_sum / total_documents_comments
            print(f"  Topic {topic_num}: Average Probability {prob_avg:.4f}")
            print("    Top words:", ", ".join([word for word, _ in lda_model_comments.show_topic(topic_num)]))
        print()

    print()

    # Method 2: LSA Latent Semantic Analysis
    print("LSA METHOD:")
    for scenario, highlights in scenario_highlights.items():
        preprocessed_highlights = preprocess_text_LSA(highlights)
        preprocessed_comments = preprocess_text_LSA(scenario_comments[scenario])

        # Vectorize text data using TF-IDF separately for highlights and comments
        vectorizer_highlights = TfidfVectorizer()
        vectorizer_comments = TfidfVectorizer()

        # Train separate LSA models for highlights and comments
        num_components = 3
        lsa_model_highlights = TruncatedSVD(n_components=num_components, random_state=42)
        lsa_model_comments = TruncatedSVD(n_components=num_components, random_state=42)

        print(f"Scenario: {scenario}")

        # Get top words or themes for scenario highlights
        print("Top words for scenario highlights:")
        tfidf_matrix_highlights = vectorizer_highlights.fit_transform(preprocessed_highlights)
        lsa_matrix_highlights = lsa_model_highlights.fit_transform(tfidf_matrix_highlights)
        feature_names_highlights = vectorizer_highlights.get_feature_names_out()
        for i in range(3):  # Print three themes per scenario highlights
            print(f"Theme {i + 1}: {', '.join(feature_names_highlights[idx] for idx in lsa_model_highlights.components_[i].argsort()[-8:][::-1])}")
        print()

        # Get top words or themes for scenario comments
        print("Top words for scenario comments:")
        tfidf_matrix_comments = vectorizer_comments.fit_transform(preprocessed_comments)
        lsa_matrix_comments = lsa_model_comments.fit_transform(tfidf_matrix_comments)
        feature_names_comments = vectorizer_comments.get_feature_names_out()
        for i in range(3):  # Print three themes per scenario comments
            print(f"Theme {i + 1}: {', '.join(feature_names_comments[idx] for idx in lsa_model_comments.components_[i].argsort()[-8:][::-1])}")
        print()
