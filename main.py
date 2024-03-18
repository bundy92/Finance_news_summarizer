from typing import List, Dict
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import re
import torch
import csv

device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Setting up the summarization model
model_name: str = "human-centered-summarization/financial-summarization-pegasus"
tokenizer: PegasusTokenizer = PegasusTokenizer.from_pretrained(model_name)
model: PegasusForConditionalGeneration = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
print("Setting up the model.")


class StockNewsSummarizer:
    def __init__(self, monitored_tickers: List[str], exclude_list: List[str]):
        self.monitored_tickers: List[str] = monitored_tickers
        self.exclude_list: List[str] = exclude_list

    def search_for_stock_news_urls(self, ticker: str) -> List[str]:
        search_url: str = f"https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws"
        r: requests.Response = requests.get(search_url, cookies={"CONSENT": "YES+cb.20210720-07-p0.en+FX+410"})
        soup: BeautifulSoup = BeautifulSoup(r.text, 'html.parser')
        atags: List[BeautifulSoup] = soup.find_all('a')
        hrefs: List[str] = [link['href'] for link in atags]
        return hrefs

    def strip_unwanted_urls(self, urls: List[str]) -> List[str]:
        val: List[str] = []
        for url in urls:
            if 'https://' in url and not any(exclude_word in url for exclude_word in self.exclude_list):
                res: str = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                val.append(res)
        return list(set(val))

    def scrape_and_process(self, URLs: List[str]) -> List[str]:
        ARTICLES: List[str] = []
        for url in URLs:
            r: requests.Response = requests.get(url)
            soup: BeautifulSoup = BeautifulSoup(r.text, 'html.parser')
            paragraphs: List[BeautifulSoup] = soup.find_all('p')
            text: List[str] = [paragraph.text for paragraph in paragraphs]
            words: str = ' '.join(text).split(' ')[:200]
            ARTICLE: str = ' '.join(words)
            ARTICLES.append(ARTICLE)
        return ARTICLES

    def summarize(self, articles: List[str]) -> List[str]:
        summaries: List[str] = []
        for article in articles:
            input_ids: torch.Tensor = tokenizer.encode(article, return_tensors='pt').to(device)
            output: torch.Tensor = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
            summary: str = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries

    def create_output_array(self, summaries: Dict[str, List[str]],
                            urls: Dict[str, List[str]]) -> List[List[str]]:
        output: List[List[str]] = []
        for ticker in self.monitored_tickers:
            for counter in range(len(summaries[ticker])):
                output_this: List[str] = [
                    ticker,
                    summaries[ticker][counter],
                    urls[ticker][counter]
                ]
                output.append(output_this)
        return output


# Building a news and sentiment pipeline
monitored_tickers: List[str] = ['TSLA', 'BTC', 'AAPL', 'AMZN']
exclude_list: List[str] = ['maps', 'policies', 'preferences', 'accounts', 'support']

stock_news_summarizer: StockNewsSummarizer = StockNewsSummarizer(monitored_tickers, exclude_list)

raw_urls: Dict[str, List[str]] = {ticker: stock_news_summarizer.search_for_stock_news_urls(ticker)
                                   for ticker in monitored_tickers}

cleaned_urls: Dict[str, List[str]] = {ticker: stock_news_summarizer.strip_unwanted_urls(raw_urls[ticker])
                                       for ticker in monitored_tickers}

articles: Dict[str, List[str]] = {ticker: stock_news_summarizer.scrape_and_process(cleaned_urls[ticker])
                                   for ticker in monitored_tickers}

summaries: Dict[str, List[str]] = {ticker: stock_news_summarizer.summarize(articles[ticker])
                                    for ticker in monitored_tickers}


print("Creating final output.")
final_output: List[List[str]] = stock_news_summarizer.create_output_array(summaries, cleaned_urls)

final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])

with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)
