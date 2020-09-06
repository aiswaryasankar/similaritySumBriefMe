# importing dependencies
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output, State
import sd_material_ui
from newspaper import Article
import gensim
from gensim.summarization import summarize
from dash.exceptions import PreventUpdate
from newspaper import fulltext
import requests
import pandas as pd
import yake
import nltk
from newsapi import NewsApiClient


leftSources = ["cnn", "buzzfeed", "the-washington-post", "bbc-news", "vice-news", "newsweek", "techcrunch", "reuters", "politico", "newsweek", "msnbc"]
rightSources = ["fox-news", "national-review", "new-york-magazine", "breitbart-news", "business-insider", "the-wall-street-journal", "bloomberg", "the-washington-times", "the-hill", "the-american-conservative"]

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

# helper functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app.layout = html.Div([

    html.Div(html.H3("Brief.Me"), style={'font-weight':'bold','background-color':'darkorange', 'color':'white','text-align':'center'}),

    html.Br(),
    html.Br(),

    dbc.Row([

        dbc.Col(dbc.Input(id='url', type='url', size=30, placeholder="Type or copy/paste an URL"), width={'size':6, 'order':1, 'offset':3}),
        dbc.Col(dbc.Button("Summarize", id='button', n_clicks=1, color="primary", className="mr-1"), width={'order':2})

        ]),

    html.Br(),

    dbc.Row([

        dbc.Col(dcc.Loading(html.Div(html.Div(id="summary"), style={'font-weight':'bold'})), width={'size':6, 'offset':3})

    ]),

    html.Div(id='table')

    # html.Div(generate_table(allData))

    # html.Table([
    #     html.Tr([html.Td([html.H1(id='Title1'), html.H2(id='Source1'), html.Br(), html.P(id='Summary1')])]),
    #     html.Tr([html.Td([html.H1(id='Title2'), html.H2(id='Source2'), html.Br(), html.P(id='Summary2')])]),
    #     html.Tr([html.Td([html.H1(id='Title3'), html.H2(id='Source3'), html.Br(), html.P(id='Summary3')])]),
    # ]),
    ],
)

def fetch_similar_articles(keyword):

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    newsapi = NewsApiClient(api_key='ce7482cbd40f4d90a8eea404e7702db6')

    top_headlines = newsapi.get_top_headlines(q=keyword,
                                              sources='bbc-news,the-wall-street-journal,the-washington-post,fox-news,bloomberg, vice-news, politico, reuters, the-hill',
                                              language='en')

    return top_headlines["articles"]

def fetch_article_text(url):
  try:
    article = Article(url)
    article.download()
    article.parse()
    return article.text
  except:
    return None

def summarizer(url):
    global allData

    leftSummaries, rightSummaries = {}, {}
    text = fetch_article_text(url)
    main_summary = summarize(text)

    keywords = extract_keywords(text)
    urls = []

    rightData, leftData, allData = get_articles_content(keywords)
    rightDf, leftDf = pd.DataFrame(rightData), pd.DataFrame(leftData)
    allSources = pd.concat([rightDf, leftDf], axis=1)

    return main_summary, allData

def get_articles_content(keywords):
    '''
    This function will return a row of the dataframe where there is a title, source, url and summary. 
    '''
    allResults, leftRows, rightRows = [], [], []

    for keyword in keywords:

        articleList = fetch_similar_articles(keyword)
        print('number of similar articles fetched')
        print(len(articleList))
        #getSimilarArticlesURL += keyword
        #dictRes = requests.get(getSimilarArticlesURL).content.decode('utf-8')

        for elem in articleList:
            source = elem['source']
            url = elem['url']
            title = elem['title']
            text = fetch_article_text(url)

            if text is not None and len(text) > 1:
                summary = summarize(text)

                allResults.append({'title': title, 'url': url,'source': source, 'summary': summary})
                if source in leftSources:
                    # leftRows.append([title, url, source, summary])
                    leftRows.append(pd.DataFrame({'title': title, 'url': url,'source': source, 'summary': summary}))
                elif source in rightSources:
                    # rightRows.append([title, url, source, summary])
                    rightRows.append(pd.DataFrame({'title': title, 'url': url, 'source': source, 'summary': summary}))

    print("ALl results dataframe")
    print(pd.DataFrame(allResults))
    allResults = pd.DataFrame(allResults)
    return leftRows, rightRows, allResults

def extract_keywords_yake(text, phrase_length, num_keywords):

  custom_kw_extractor = yake.KeywordExtractor(n=phrase_length, top=num_keywords)
  keywords = custom_kw_extractor.extract_keywords(text)

  return keywords

def extract_keywords(text):
    '''
    Returns a list of keywords given the article text. 
    '''
    global getKeywordsURL

    getKeywordsURL += text
    keywordRes = extract_keywords_yake(text, 2, 5)
    keywords = []

    for pair in keywordRes:
        keywords.append(pair[1])

    return keywords


@app.callback(
    [Output('summary', 'children'),
    Output('table', 'children')],
    [Input("button", "n_clicks")], state=[State('url', 'value')])
def update_table(n_click:int, url):
    if n_click>1:
        summary, table = summarizer(url)
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in table.columns
        ]
        table = table.to_dict('records')
        # return table, columns
        return summary, dt.DataTable(data=table, columns=columns)
    else: 
        return [], []

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)






