# from bs4 import BeautifulSoup
# import urllib.request
# import csv
# from selenium import webdriver


# # Scrape CNN
# def scrape_CNN(keyword):
# 	cnn_url = "https://www.cnn.com/search?q="
# 	# page = urllib.request.urlopen(cnn_url)
# 	# soup = BeautifulSoup(page, 'html.parser')
# 	# print("CNN soup")
# 	# print(soup.prettify())
# 	browser = webdriver.Chrome("/Users/aiswarya.s/Downloads/chromedriver")
# 	url = "https://www.nytimes.com/search?query="
# 	browser.get(url) #navigate to the page
# 	innerHTML = browser.execute_script("return document.body.innerHTML")
# 	print(innerHTML)


# # Scrape New York Times
# def scrape_NYT(keyword):
# 	nyt_url = "https://www.nytimes.com/search?query="
# 	page = urllib.request.urlopen(nyt_url)
# 	soup = BeautifulSoup(page, 'html.parser')
# 	soup.find("")
# 	print(soup.prettify())

# # Scrape Fox News
# def scrape_Fox(keyword):
# 	fox_url = "https://www.foxnews.com/search-results/search?q="
# 	page = urllib.request.urlopen(fox_url)
# 	soup = BeautifulSoup(page, 'html.parser')
# 	print("Fox soup")
# 	print(soup.prettify())

# # Scrape Washington Post
# def scrape_WashingtonPost(keyword):
# 	wapo_url = "https://www.washingtonpost.com/newssearch/?query="
# 	page = urllib.request.urlopen(wapo_url)
# 	soup = BeautifulSoup(page, 'html.parser')
# 	print("Wapo soup")
# 	print(soup.prettify())

# def web_scraper(keywords):
# 	"""
# 		Given the provided url, go ahead and return a list of articles in the following format:
# 		[articleURL, title, text, source]
# 	"""
# 	allArticles = []

# 	for keyword in keywords:
# 		allArticles.append(scrape_CNN(keyword))
# 		# allArticles.append(scrape_NYT(keyword))
# 		# allArticles.append(scrape_Fox(keyword))
# 		# allArticles.append(scrape_WashingtonPost(keyword))

# 		print(allArticles)

# web_scraper(["amy"])

