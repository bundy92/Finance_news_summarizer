# Stock News Summarizer

This Python script is designed to summarize news articles related to specific stock tickers using the Pegasus model for conditional generation. It scrapes news articles from Google search results, processes them, and generates summaries.

## Requirements

- Python 3.x
- `transformers` library
- `bs4` (Beautiful Soup) library
- `requests` library
- `torch` library
- Stock ticker symbols to monitor

## Setup

1. Install the required libraries using pip

   ```bash
   pip install transformers beautifulsoup4 requests torch
   ```

2. Ensure that CUDA is available if you want to run the script on GPU.

## Usage

1. Instantiate the `StockNewsSummarizer` class with a list of monitored stock tickers and an exclude list for filtering URLs.

   ```python
   monitored_tickers = ['TSLA', 'BTC', 'AAPL', 'AMZN']
   exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
   stock_news_summarizer = StockNewsSummarizer(monitored_tickers, exclude_list)
   ```

2. Run the script. It will perform the following steps:
   - Search for news URLs related to the monitored tickers.
   - Clean the URLs by removing unwanted words and duplicates.
   - Scrape and process the articles from the cleaned URLs.
   - Generate summaries for the articles.
   - Create a final output array containing the ticker, summary, and URL.
   - Write the output to a CSV file named `assetsummaries.csv`.

## Notes

- The `create_output_array` method in the `StockNewsSummarizer` class creates the final output array by combining summaries and URLs.

- The script writes the output to a CSV file for further analysis.

## Example

```python
# Instantiate StockNewsSummarizer
monitored_tickers = ['TSLA', 'BTC', 'AAPL', 'AMZN']
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
stock_news_summarizer = StockNewsSummarizer(monitored_tickers, exclude_list)

# Run the script
stock_news_summarizer.run()
```

## Output

The script generates a CSV file named `assetsummaries.csv`, containing summarized news articles for the monitored tickers.

For any questions or issues, please contact @bundy92.
