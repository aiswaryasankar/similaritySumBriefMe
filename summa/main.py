import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output, State
import sd_material_ui
from newspaper import Article
import gensim
import gensim.summarization
from dash.exceptions import PreventUpdate
from newspaper import fulltext
import requests
import pandas as pd
import yake
import nltk
from newsapi import NewsApiClient
from summarizer import summarize
#from web_scraper import webscrapper
import csv
import spacy

nlp = spacy.load("en_core_web_sm")


leftSources = ["cnn", "buzzfeed", "the-washington-post", "bbc-news", "vice-news", "newsweek", "techcrunch", "reuters", "politico", "newsweek", "msnbc"]
rightSources = ["fox-news", "national-review", "new-york-magazine", "breitbart-news", "business-insider", "the-wall-street-journal", "bloomberg", "the-washington-times", "the-hill", "the-american-conservative"]

validSources = ["cnn", "buzzfeed", "the-washington-post", "bbc-news", "vice-news", "newsweek", "techcrunch", "reuters", "politico", "newsweek", "msnbc", "fox-news", "national-review", "new-york-magazine", "breitbart-news", "business-insider", "the-wall-street-journal", "bloomberg", "the-washington-times", "the-hill", "the-american-conservative"]
validSourcesContextual = ["npr", "latimes", "washingtontimes", "thenation", "forbes", "realclearpolitics", "cnn", "cnnnext", "independent", "nytimes", "nypost", "nationalpost", "dailyreview", "cbsnews", "americanindependent", "fox-news", "nymag", "propublica", "politico", "fivethirtyeight", ]

# importing CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# similarArticleURL
getSimilarArticlesURL = "https://us-central1-secure-site-266302.cloudfunctions.net/getSimilarArticles?keywords="
getKeywordsURL = "https://us-central1-secure-site-266302.cloudfunctions.net/getKeyword?text="
getArticleTextURL = "https://us-central1-secure-site-266302.cloudfunctions.net/getArticleText?url="

allData = pd.DataFrame()


# instantiating dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # the flask app

SENTENCE = 0
WORD = 1
TITLE_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.01

DEFAULT_RATIO = 0.1

app.layout = html.Div([

    html.Br(),
    html.Br(),
    html.Br(),

    html.Div(html.H1("Brief.Me"), style={'font-weight':'bold', 'color':'darkblue','text-align':'center'}),

    html.Br(),
    html.Br(),

    dbc.Row([

        dbc.Col(dbc.Input(id='url', type='url', size=30, placeholder="Type or copy/paste an URL"), width={'size':6, 'order':1, 'offset':3}),
        dbc.Col(dbc.Button("Summarize", id='button', n_clicks=1, color="primary", className="mr-1"), width={'order':2})

        ]),

    html.Br(),
    html.Br(),
    html.Br(),

    dbc.Row([

        dbc.Col(dcc.Loading(html.Div(html.H3(id="mainTitle"), style={'text-align':'center','font-weight':'bold'})), width={'size':6, 'offset':3})

    ]),

    html.Br(),
    html.Br(),

    dbc.Col(html.Div(html.Div(id="mainSummary"), style={'font-weight':'bold'}), width={'size':6, 'offset':3}),

    html.Br(),
    html.Br(),
    html.Br(),

    dbc.Row([
        dbc.Col(html.H4(id='title1'), width={'size':5, 'offset':1}, style={'text-align':'center'}),
        dbc.Col(html.H4(id='title2'), width={'size':5, 'offset':0}, style={'text-align':'center'}),
        ]),

    dbc.Row([
        dbc.Col(html.H6(id='source1'), width={'size':5, 'offset':1}, style={'text-align':'center'}),
        dbc.Col(html.H6(id='source2'), width={'size':5, 'offset':0}, style={'text-align':'center'}),
        ]),

    html.Br(),
    dbc.Row([
        dbc.Col(html.H6(id='summary1'), width={'size':5, 'offset':1}),
        dbc.Col(html.H6(id='summary2'), width={'size':5, 'offset':0}),
        ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H6(id='score1'), width={'size':5, 'offset':1}, style={'text-align':'left'}),
        dbc.Col(html.H6(id='score2'), width={'size':5, 'offset':0}, style={'text-align':'left'}),
        ]),
    html.Br(),

    html.Br(),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H4(id='title3'), width={'size':5, 'offset':1}, style={'text-align':'center'}),
        dbc.Col(html.H4(id='title4'), width={'size':5, 'offset':0}, style={'text-align':'center'}),
        ]),

    dbc.Row([
        dbc.Col(html.H6(id='source3'), width={'size':5, 'offset':1}, style={'text-align':'center'}),
        dbc.Col(html.H6(id='source4'), width={'size':5, 'offset':0}, style={'text-align':'center'}),
        ]),

    html.Br(),
    dbc.Row([
        dbc.Col(html.H6(id='summary3'), width={'size':5, 'offset':1}),
        dbc.Col(html.H6(id='summary4'), width={'size':5, 'offset':0}),
        ]),

    html.Br(),
    dbc.Row([
        dbc.Col(html.H6(id='score3'), width={'size':5, 'offset':1}, style={'text-align':'left'}),
        dbc.Col(html.H6(id='score4'), width={'size':5, 'offset':0}, style={'text-align':'left'}),
        ]),
    html.Br(),

    html.Br(),
    dbc.Row([
        dbc.Col(html.H6(id='url1'), width={'size':5, 'offset':1}, style={'text-align':'left'}),
        dbc.Col(html.H6(id='url2'), width={'size':5, 'offset':0}, style={'text-align':'left'}),
        ]),
        html.Br(),
    dbc.Row([
        dbc.Col(html.H6(id='url3'), width={'size':5, 'offset':1}, style={'text-align':'left'}),
        dbc.Col(html.H6(id='url4'), width={'size':5, 'offset':0}, style={'text-align':'left'}),
        ]),
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(dt.DataTable(id="table"))))
    ]),
])

def fetch_similar_articles(keyword):

    newsapi = NewsApiClient(api_key='ce7482cbd40f4d90a8eea404e7702db6')

    top_headlines = newsapi.get_top_headlines(q=keyword,
                                              sources='bbc-news,the-wall-street-journal,the-washington-post,fox-news,bloomberg, vice-news, politico, reuters, the-hill',
                                              language='en', page_size=5)

    return top_headlines["articles"]

def fetch_article_text_and_title(url):
  try:
    article = Article(url)
    article.download()
    article.parse()
    return article.text, article.title
  except:
    return None, None

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def fetch_similar_articles_contextual(query):

    url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"

    querystring = {"toPublishedDate":"null","fromPublishedDate":"null","pageSize":"10","q":query,"autoCorrect":"false","pageNumber":"1"}

    headers = {
        'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
        'x-rapidapi-key': "346ba7f35fmshf502c0d01ac7d9cp1dd4cbjsncdb8fdba99e3"
        }

    response = requests.request("GET", url, headers=headers, params=querystring).json()

    articleList = []
    allUrls = set()

    for result in response["value"]:
        if result["title"] not in allUrls:
            title = remove_html_tags(result["title"])
            articleList.append({'source': result["provider"]["name"], 'title': title, 'url': result["url"]})
        allUrls.add(result["title"])

    return articleList

def rank_articles(mainArticle, relatedArticles):
    """
    This method should go ahead and compute the similarity scores between each article summary and the main article summary and return the appropriate summaries as well. It will then rank the summaries by the score that is generated. It will call textRank with text1, text2.
    """
    ranking = []
    mainArticleText = mainArticle["text"]
    print("number of articles to rank: " + str(len(relatedArticles)))

    for article in relatedArticles:
        print(article)
        articleText = article['text']
        if articleText == None:
            continue

        print(textrank(mainArticle, article, words=250))
        summary, score = textrank(mainArticle, article)
        if score == [] or summary == []:
            continue
            
        article["summary"] = summary
        article["score"] = score
        ranking.append(article)

    # The output of this needs to be the structure of the project
    # It has to be the same article structure 
    print("ranking")
    print(ranking)
    sortedArticles = sorted(ranking, key=lambda k: k['score'], reverse=True) 

    # with open('rankings.csv', mode='a') as rankings:
    #     fieldnames = ['summary', 'score', 'source', 'text', 'title', 'url']
    #     writer = csv.DictWriter(rankings, fieldnames=fieldnames)
    #     writer.writerow(mainArticle["url"])
    #     for data in sortedArticles:
    #         writer.writerow(data)

    return sortedArticles

def summarizer(url):

    global allData

    text, title = fetch_article_text_and_title(url)

    try:
        main_summary = gensim.summarization.summarize(text, word_count=150)
    except:
        main_summary = text

    keywords = extract_keywords(text, title)
    urls = []
    mainArticle = {"url": url, "text": text, "title": title}
    allArticles = get_related_articles(keywords)
    rankedArticles = rank_articles(mainArticle, allArticles)

    rankedArticles = pd.DataFrame(rankedArticles)
    return title, main_summary, rankedArticles


def textrank(article1, article2, words=250):
    return summarize(article1, article2, words)


def hydrate_article_entity(article):
    """
    This method returns a standard article representation.
    """

    articleEntity = {}
    text, title = fetch_article_text_and_title(article["url"])
    if text == '':
        return {}

    articleEntity["url"] = article["url"]
    articleEntity["title"] = title
    articleEntity["source"] = article['source']
    articleEntity['text'] = text

    return articleEntity


def get_related_articles(keywords):
    """
    This function returns all the related articles for a list of keywords
    [article: {"url": url, "title": title, "Source": source, "text": text}]
    """
    relatedArticleUrls = set()
    relatedArticles = []

    for keyword in keywords:
        newsApiArticles = fetch_similar_articles(keyword)
        for article in newsApiArticles:
            if article['url'] not in relatedArticleUrls:
                articleEntity = hydrate_article_entity(article)
                if articleEntity != {}:
                    relatedArticles.append(articleEntity)
                    relatedArticleUrls.add(article['url'])

        contextualArticles = fetch_similar_articles_contextual(keyword)
        for article in contextualArticles:
            if article['url'] not in relatedArticleUrls:
                articleEntity = hydrate_article_entity(article)
                if articleEntity != {}:
                    relatedArticles.append(articleEntity)
                    relatedArticleUrls.add(article['url'])

    print("number of similar articles fetched: " + str(len(relatedArticles)))
    return relatedArticles

def extract_keywords_yake(text, phrase_length, num_keywords):

  custom_kw_extractor = yake.KeywordExtractor(n=phrase_length, top=num_keywords)
  keywords = custom_kw_extractor.extract_keywords(text)

  valid_keywords = []
  print(keywords)

  return keywords

def extract_keywords(text, title):
    '''
    Returns a list of keywords given the article text. 
    '''
    global getKeywordsURL

    # Extract nouns from title
    # labels = nlp(title)
    # for ent in labels.ents:
    #     print('entity')
    #     print([ent.text, ent.label_])
    #     if ent.label_ == 'NORP':
    #         text, pos = ent

    getKeywordsURL += text
    keywordResTitle = extract_keywords_yake(title, 2, 2)
    print(keywordResTitle)
    # keywordResTitle = [(score, phrase) for score, phrase in keywordResTitle if score > TITLE_THRESHOLD]

    keywordResText = extract_keywords_yake(text, 2, 3)
    print(keywordResText)
    # keywordResText = [(score, phrase) for score, phrase in keywordResText if score > TEXT_THRESHOLD]

    keywordsRes = list(set(keywordResText + keywordResTitle))
    keywords = []

    print("title keywords")
    print(keywordResTitle)
    print("text keywords")
    print(keywordResText)

    for pair in keywordsRes:
        keywords.append(pair[1])

    print(keywords)
    return keywords


@app.callback( 
    [Output('mainTitle', 'children'),
    Output('mainSummary', 'children'),
    Output('title1', 'children'),
    Output('source1', 'children'),
    Output('summary1', 'children'),
    Output('score1', 'children'),
    Output('url1', 'children'),
    Output('title2', 'children'),
    Output('source2', 'children'),
    Output('summary2', 'children'),
    Output('score2', 'children'),
    Output('url2', 'children'),
    Output('title3', 'children'),
    Output('source3', 'children'),
    Output('summary3', 'children'),
    Output('score3', 'children'),
    Output('url3', 'children'),
    Output('title4', 'children'),
    Output('source4', 'children'),
    Output('summary4', 'children'),
    Output('score4', 'children'),
    Output('url4', 'children')],
    [Input("button", "n_clicks")],
    [State('url', 'value')])
def update_table(n_click:int, url):
    if n_click>1:

        mainTitle, mainSummary, table = summarizer(url)
        columns=[{"name": i, "id": i, "deletable": True, "selectable": True} for i in table.columns]
        table = table.to_dict('records')

        if len(table) >= 4:
            title1 = table[0]['title']
            source1 = table[0]['source']
            summary1 = table[0]['summary']
            score1 = table[0]['score']
            url1 = table[0]['url']
            title2 = table[1]['title']
            source2 = table[1]['source']
            summary2 = table[1]['summary']
            score2 = table[2]['score']
            url2 = table[1]['url']
            title3 = table[2]['title']
            source3 = table[2]['source']
            summary3 = table[2]['summary']
            score3 = table[3]['score']
            url3 = table[2]['url']
            title4 = table[3]['title']
            source4 = table[3]['source']
            summary4 = table[3]['summary']
            score4 = table[3]['score']
            url4 = table[3]['url']

            return mainTitle, mainSummary, title1, source1, summary1, score1, url1, title2, source2, summary2, score2, url2, title3, source3, summary3, score3, url3, title4, source4, summary4, score4, url4

        else:
            return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    else: 
        return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)




