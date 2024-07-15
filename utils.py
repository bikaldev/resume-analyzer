import os

import requests
from bs4 import BeautifulSoup

def get_user_repos(username):
    # GitHub API URL for user repos

    # Define  token and headers
    try:
        token = os.environ['GITHUB_ACCESS_TOKEN'] 
    except KeyError as e:
        raise Exception('Environment variable GITHUB_ACCESS_TOKEN must be set.')
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    repos_url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(repos_url, headers=headers)
    repos_data = response.json()

    if response.status_code == 200:
        all_repos_info = []

        for repo in repos_data:
            repo_info = get_repo_info(username, repo['name'])
            if('error' not in repo_info):
                all_repos_info.append(repo_info)
        
        return all_repos_info
    else:
        return {'error': 'User not found or no repositories'}

def get_repo_info(owner, repo):
    # GitHub API URL for repo details
    api_url = f'https://api.github.com/repos/{owner}/{repo}'
    response = requests.get(api_url)
    repo_data = response.json()

    if response.status_code == 200:
        # Fetch repository details from GitHub API
        repo_title = repo_data['name']
        stars = repo_data['stargazers_count']
        
        # Fetch repository page for additional information
        html_url = f'https://github.com/{owner}/{repo}'
        html_response = requests.get(html_url)
        soup = BeautifulSoup(html_response.content, 'html.parser')

        # Extracting text in the About section
        about_section = soup.find('p', {'class': 'f4 my-3'})
        about_text = about_section.text.strip() if about_section else None

        # Extracting tools and languages used
        tools = []
        for lang_tag in soup.find_all('span', class_='color-fg-default text-bold mr-1'):
            tools.append(lang_tag.text.strip())

        return {
            'title': repo_title,
            'about': about_text,
            'tools': tools,
            'stars': stars
        }
    else:
        return {'error': 'Repository not found'}


config = {
    "BOT_NAME": "webcrawler",
    "SPIDER_MODULES": ["webcrawler.webcrawler.spiders"],
    "NEWSPIDER_MODULE": "webcrawler.webcrawler.spiders",
    "ROBOTSTXT_OBEY": True,
    "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
    "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
    "FEED_EXPORT_ENCODING": "utf-8",
    "SPIDER_MIDDLEWARES": {
        "scrapy.spidermiddlewares.depth.DepthMiddleware": None,
        "webcrawler.webcrawler.middlewares.DomainDepthMiddleware": 700,
    },
    "FEED_FORMAT": "json",
    "FEED_URI": "output.json",
    "DOMAIN_DEPTHS" : {'github.com': 1, 'linkedin.com': 2, 'scholar.google.com': 2},
    "DEPTH_LIMIT": 2,
    "LOG_LEVEL": "WARNING",  # Reduce the verbosity
}

# from scrapy.crawler import CrawlerProcess
# from webcrawler.webcrawler.spiders.myspider import MySpider

# def run_spider(domain):
#     process = CrawlerProcess(settings=config)
#     process.crawl(MySpider, domain=domain)
#     process.start()