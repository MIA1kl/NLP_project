import requests
from bs4 import BeautifulSoup
import json
import re
import time

# Function to retrieve HTML content from a URL
def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    return response.content

# Function to remove HTML tags and source markup from text
def clean_text(text):
    # Remove HTML tags using regular expressions
    clean_text = re.sub('<.*?>', '', text)
    
    # Remove other source markup if necessary
    
    return clean_text.strip()

# Function to extract article information from HTML
def extract_article_info(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    articles = soup.find_all('div', class_='latest-summary')
    article_info_list = []
    
    for article in articles:
        title_element = article.find('h3')
        link_element = article.find('a')
        summary_element = article.find('p')
        
        title = title_element.text.strip() if title_element else ''
        link = link_element['href'] if link_element else ''
        summary = summary_element.text.strip() if summary_element else ''
        
        article_info_list.append({
            'title': title,
            'link': link,
            'summary': summary
        })
    
    return article_info_list

# Function to crawl the website and extract articles
def crawl_website(url):
    html = get_html(url)
    article_info_list = extract_article_info(html)
    
    for article_info in article_info_list:
        time.sleep(2)  # Pause between requests to be respectful to the website
        
        article_html = get_html(article_info['link'])
        # Extract additional information from the article HTML as needed
        
        # Clean and process the extracted information
        article_text = clean_text(article_html)
        
        # Store the information in JSON format or perform any other desired operations
        # Example:
        data = {
            'title': article_info['title'],
            'summary': article_info['summary'],
            'text': article_text,
            'link': article_info['link']
            # Add more attributes as needed
        }
        
        # Store the data in a JSON file
        file_name = article_info['link'].split('/')[-1] + '.json'
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Article saved: {file_name}")

# Start crawling the website
crawl_website('https://www.sciencedaily.com/news/computers_math/computer_science')