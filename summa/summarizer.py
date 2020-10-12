from math import log10

from pagerank_weighted import pagerank_weighted_scipy as _pagerank
from preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from commons import build_graph as _build_graph
from commons import remove_unreachable_nodes as _remove_unreachable_nodes
from syntactic_unit import SyntacticUnit
import pandas as pd
import scipy

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

def _set_graph_edge_weights(graph):
    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():

            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):

                similarity = _get_similarity(sentence_1, sentence_2)
                if similarity != 0:
                    graph.add_edge(edge, similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _get_similarity(s1, s2):
    if s1 == [] or s2 == []:
        return 0

    if type(s1) is list and type(s1[0]) is SyntacticUnit:
        s1words = ""
        for unit in s1:
            s1words += unit.token
        s1 = s1words

    if type(s2) is list and type(s2[0]) is SyntacticUnit:
        s2words = ""
        for unit in s2:
            s2words += unit.token
        s2 = s2words

    words_sentence_one = s1.split()
    words_sentence_two = s2.split()

    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)

def _get_similarity_bert(s1, s2):
    sentences = [s1, s2]
    sentence_embeddings = model.encode(sentences)  
    distance = scipy.spatial.distance.cdist([sentence_embeddings[0]], [sentence_embeddings[1]], "cosine")[0]
    return distance


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _format_results(extracted_sentences, split, score):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _add_scores_to_sentences(sentences, scores):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def _extract_most_important_sentences(sentences, words, sourceNum):
    releventSentences = []
    for sentence in sentences:
        if sentence.source == sourceNum:
            releventSentences.append(sentence)

    releventSentences.sort(key=lambda s: s.score, reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio.
    if words is None:
        length = len(releventSentences) * ratio
        return releventSentences[:int(length)]

    # Else, the ratio is ignored.
    else:
        return _get_sentences_with_word_count(releventSentences, words)

def summaryText(summary):
    summaryText = ""
    for unit in summary:
        summaryText += (" " + unit.text)
    return summaryText

def getDifferences(summary1, summary2):
    diff = []
    for sentence in summary1:
        if sentence not in summary2:
            diff.append(sentence)
    return diff


def summarize(article1, article2, words=250, language="english", split=False, scores=False, additional_stopwords=None):

    text1=article1["text"]
    text2=article2["text"]

    if not isinstance(text1, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")
    if not isinstance(text2, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # Gets a list of processed sentences.
    # Each sentence is created as a syntactic unit with the source article number to be referenced later on as well
    languague="english"
    sentencesText1 = _clean_text_by_sentences(text1, source=0, additional_stopwords=additional_stopwords)
    sentencesText2 = _clean_text_by_sentences(text2, source=1, additional_stopwords=additional_stopwords)
    allSentences = _clean_text_by_sentences(text1, source=0, additional_stopwords=additional_stopwords) + _clean_text_by_sentences(text2, source=1, additional_stopwords=additional_stopwords)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graphCombined = _build_graph([sentence.token for sentence in allSentences])
    _set_graph_edge_weights(graphCombined)

    graph1 = _build_graph([sentence.token for sentence in sentencesText1])
    _set_graph_edge_weights(graph1)

    graph2 = _build_graph([sentence.token for sentence in sentencesText2])
    _set_graph_edge_weights(graph2)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graphCombined)
    _remove_unreachable_nodes(graph1)
    _remove_unreachable_nodes(graph2)

    # PageRank cannot be run in an empty graph.
    if len(graphCombined.nodes()) == 0:
        print("empty graph combined")
        return [], [] 
    if len(graph1.nodes()) == 0:
        print("empty graph 1")
        return [], [] 
    if len(graph2.nodes()) == 0:
        print("empty graph 2")
        return [], [] 

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores_combined = _pagerank(graphCombined)
    pagerank_scores_1 = _pagerank(graph1)
    pagerank_scores_2 = _pagerank(graph2)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentencesText1, pagerank_scores_1)
    _add_scores_to_sentences(sentencesText2, pagerank_scores_2)
    _add_scores_to_sentences(allSentences, pagerank_scores_combined)

    # I want to create a table that shows the scores of each sentence in the combined graph vs the summary individually and the scores attributed to it
    sentenceScores = []
    for sentence in allSentences:
        if sentence.token in pagerank_scores_1 and sentence.token in pagerank_scores_combined:
            row = {"sentence": sentence.token, "combinedScore": pagerank_scores_combined[sentence.token], "summary1Score": pagerank_scores_1[sentence.token]}
            sentenceScores.append(row)
        if sentence.token in pagerank_scores_2 and sentence.token in pagerank_scores_combined:
            row = {"sentence": sentence.token, "combinedScore": pagerank_scores_combined[sentence.token], "summary2Score": pagerank_scores_2[sentence.token]}
            sentenceScores.append(row)

    df = pd.DataFrame(sentenceScores)
    print(df)


    # Extracts the most important sentences with the selected criterion.
    # print("_____TEXT1_____")
    # for sentence in sentencesText1:
    #     print(sentence)
    #     print(sentence.score)
    # print("_____TEXT2_____")
    # for sentence in sentencesText2:
    #     print(sentence)
    #     print(sentence.score)
    # print("_____TEXT_COMBINED_____")
    # for sentence in allSentences:
    #     print(sentence)
    #     print(sentence.score)

    summary1_combined = _extract_most_important_sentences(allSentences, words, 0)
    summary2_combined = _extract_most_important_sentences(allSentences, words, 1)

    summary1Text = [unit.text for unit in summary1_combined][0]
    summary2Text = [unit.text for unit in summary2_combined][0]

    similarityScore = _get_similarity_bert(summary1Text, summary2Text)

    # print("Similarity of separate graphs: " + str(separate_graphs_similarity))
    # print("Similarity of combined graphs: " + str(combined_graphs_similarity))

    # Sorts the extracted sentences by apparition order in the original text.
    summary2_combined.sort(key=lambda s: s.index)
    summaryCombined = summaryText(summary2_combined)

    # Compute the similarity score of the similar sentences here and the sentences computed without doing the one graph approach
    return summaryCombined, similarityScore


def get_graph(text, language="english"):
    sentences = _clean_text_by_sentences(text, language)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph
