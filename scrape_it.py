# # import scrapy
# # from urllib.parse import urlparse, urljoin
# # import time
# # import random
# # import json

# # class GeneralSpider(scrapy.Spider):
# #     name = 'general'
# #     page_count = 0
# #     max_pages = 100
# #     priority_urls = set()
# #     output_file = 'website_content.json'
# #     crawled_data = []

# #     custom_settings = {
# #         'DOWNLOAD_DELAY': 5,
# #         'RANDOMIZE_DOWNLOAD_DELAY': True,
# #         'RETRY_TIMES': 5,
# #         'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408, 400],
# #     }

# #     def __init__(self, start_url=None, *args, **kwargs):
# #         super(GeneralSpider, self).__init__(*args, **kwargs)
# #         if start_url is not None:
# #             self.start_urls = [start_url]
# #             self.allowed_domains = [urlparse(start_url).netloc]
        
# #         # Clear the output file at the start of each run
# #         with open(self.output_file, 'w') as f:
# #             json.dump([], f)

# #     def get_retry_request(self, request, spider, reason):
# #         retry_times = request.meta.get('retry_times', 0) + 1
# #         backoff_delay = 2 ** retry_times
# #         random_delay = random.uniform(0, 3)
# #         total_delay = backoff_delay + random_delay
# #         time.sleep(total_delay)
# #         return request
# #     def closed(self, reason):
# #         with open(self.output_file, 'w') as f:
# #             json.dump(self.crawled_data, f, indent=4)
# #         self.logger.info(f'Saved {len(self.crawled_data)} pages to {self.output_file}')

# #     def parse(self, response):
# #         if self.page_count >= self.max_pages:
# #             return

# #         self.page_count += 1

# #         # Extract content
# #         content = {
# #             'url': response.url,
# #             'title': response.css('title::text').get(),
# #             'h1': response.css('h1::text').getall(),
# #             'h2': response.css('h2::text').getall(),
# #             'h3': response.css('h3::text').getall(),
# #             'paragraphs': response.css('p::text').getall(),
# #             # Highlight: Changed the list item selector and added string cleaning
# #             'list_items': [item.strip() for item in response.css('li::text, li *::text').getall() if item.strip()],
# #         }

# #         # Add the content to our list of crawled data
# #         self.crawled_data.append(content)

# #         self.logger.info(f'Crawled page {self.page_count}: {response.url}')

# #         # Follow links (rest of the code remains the same)
# #         if response.url == self.start_urls[0]:
# #             for href in response.css('a::attr(href)').getall():
# #                 full_url = urljoin(response.url, href)
# #                 if urlparse(full_url).netloc == self.allowed_domains[0]:
# #                     self.priority_urls.add(full_url)

# #         for href in response.css('a::attr(href)').getall():
# #             full_url = urljoin(response.url, href)
# #             if urlparse(full_url).netloc == self.allowed_domains[0]:
# #                 time.sleep(random.uniform(5, 10))
# #                 if full_url in self.priority_urls:
# #                     yield response.follow(full_url, self.parse, priority=1)
# #                 else:
# #                     yield response.follow(full_url, self.parse)
# import scrapy
# from urllib.parse import urlparse, urljoin
# import time
# import random
# import json

# class GeneralSpider(scrapy.Spider):
#     name = 'general'
#     page_count = 0
#     max_pages = 100
#     priority_urls = set()
#     output_file = 'website_content.json'
#     crawled_data = []

#     custom_settings = {
#         'DOWNLOAD_DELAY': 5,
#         'RANDOMIZE_DOWNLOAD_DELAY': True,
#         'RETRY_TIMES': 5,
#         'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408, 400],
#     }

#     def __init__(self, start_url=None, *args, **kwargs):
#         super(GeneralSpider, self).__init__(*args, **kwargs)
#         if start_url is not None:
#             self.start_urls = [start_url]
#             self.allowed_domains = [urlparse(start_url).netloc]
        
#         # Highlight: Removed the file clearing at initialization

#     def get_retry_request(self, request, spider, reason):
#         retry_times = request.meta.get('retry_times', 0) + 1
#         backoff_delay = 2 ** retry_times
#         random_delay = random.uniform(0, 3)
#         total_delay = backoff_delay + random_delay
#         time.sleep(total_delay)
#         return request

#     # Highlight: Added method to save data incrementally
#     def save_data(self):
#         with open(self.output_file, 'w') as f:
#             json.dump(self.crawled_data, f, indent=4)
#         self.logger.info(f'Saved {len(self.crawled_data)} pages to {self.output_file}')

#     def closed(self, reason):
#         self.save_data()

#     def parse(self, response):
#         if self.page_count >= self.max_pages:
#             return

#         self.page_count += 1

#         # Extract content
#         content = {
#             'url': response.url,
#             'title': response.css('title::text').get(),
#             'h1': response.css('h1::text').getall(),
#             'h2': response.css('h2::text').getall(),
#             'h3': response.css('h3::text').getall(),
#             'paragraphs': response.css('p::text').getall(),
#             'list_items': [item.strip() for item in response.css('li::text, li *::text').getall() if item.strip()],
#         }

#         # Add the content to our list of crawled data
#         self.crawled_data.append(content)

#         # Highlight: Save data incrementally every 10 pages
#         if self.page_count % 5 == 0:
#             self.save_data()

#         self.logger.info(f'Crawled page {self.page_count}: {response.url}')

#         # Follow links
#         if response.url == self.start_urls[0]:
#             for href in response.css('a::attr(href)').getall():
#                 full_url = urljoin(response.url, href)
#                 if urlparse(full_url).netloc == self.allowed_domains[0]:
#                     self.priority_urls.add(full_url)

#         for href in response.css('a::attr(href)').getall():
#             full_url = urljoin(response.url, href)
#             if urlparse(full_url).netloc == self.allowed_domains[0]:
#                 # Highlight: Removed sleep here to respect Scrapy's built-in delay
#                 if full_url in self.priority_urls:
#                     yield response.follow(full_url, self.parse, priority=1)
#                 else:
#                     yield response.follow(full_url, self.parse)
import scrapy
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError, TimeoutError
from scrapy.utils.response import get_base_url
from urllib.parse import urljoin
import json
import time
import random
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message


class GeneralSpider(scrapy.Spider):
    max_pages = 1000  # Set a maximum number of pages to crawl
    pages_crawled = 0
    name = 'robust_general'
    output_file = 'website_content.json'  # Add this line

    page_count = 0
    custom_settings = {
    'DOWNLOAD_DELAY': 10,  # Increased from 5 to 10
    'RANDOMIZE_DOWNLOAD_DELAY': True,
    'CONCURRENT_REQUESTS': 1,  # Reduced concurrent requests
    'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # Added to limit requests per domain
    'RETRY_TIMES': 5,
    'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408, 400],
    }

    def __init__(self, start_url=None, *args, **kwargs):
        super(GeneralSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url] if start_url else []
        self.crawled_data = []
        self.crawled_urls = set()

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, errback=self.errback_httpbin,
                                 meta={'download_timeout': 30})

    def parse(self, response):
        
        if self.pages_crawled >= self.max_pages:
            self.logger.info(f"Reached maximum number of pages ({self.max_pages}). Stopping.")
            return

        self.pages_crawled += 1

        url = response.url
        if url in self.crawled_urls:
            return
        self.crawled_urls.add(url)

        content = {
            'url': url,
            'title': response.css('title::text').get(),
            'text_content': self.extract_text_content(response),
        }
        if self.page_count % 5 == 0:
            self.save_partial_results()
        self.crawled_data.append(content)
        self.save_partial_results()

        # Follow links
        base_url = get_base_url(response)
        for href in response.css('a::attr(href)').getall():
            full_url = urljoin(response.url, href)
            if full_url.startswith('http'):
                time.sleep(random.uniform(5, 15))  # Added delay between requests
                yield scrapy.Request(full_url, callback=self.parse, errback=self.errback_httpbin,
                                     meta={'download_timeout': 30}, dont_filter=True)

    def extract_text_content(self, response):
        # Combine multiple selectors to get more content
        selectors = [
            'p::text', 'h1::text', 'h2::text', 'h3::text', 'li::text',
            'div::text', 'span::text', 'article::text'
        ]
        text_content = []
        for selector in selectors:
            text_content.extend(response.css(selector).getall())
        
        # Use XPath as a fallback
        if not text_content:
            text_content = response.xpath('//body//text()').getall()

        return ' '.join([text.strip() for text in text_content if text.strip()])

    def errback_httpbin(self, failure):
        if failure.check(HttpError):
            response = failure.value.response
            self.logger.error(f'HttpError on {response.url}')
        elif failure.check(DNSLookupError):
            request = failure.request
            self.logger.error(f'DNSLookupError on {request.url}')
        elif failure.check(TimeoutError):
            request = failure.request
            self.logger.error(f'TimeoutError on {request.url}')

    def save_partial_results(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.crawled_data, f, indent=4)
        self.logger.info(f'Saved {len(self.crawled_data)} pages to {self.output_file}')
    def closed(self, reason):
        with open('final_results.json', 'w') as f:
            json.dump(self.crawled_data, f, indent=4)

class CustomRetryMiddleware(RetryMiddleware):
    def __init__(self, settings):
        super().__init__(settings)
        self.max_retry_times = settings.getint('RETRY_TIMES')

    def process_response(self, request, response, spider):
        if request.meta.get('dont_retry', False):
            return response
        if response.status in self.retry_http_codes:
            retries = request.meta.get('retry_times', 0) + 1
            if retries <= self.max_retry_times:
                retryreq = self._retry(request, reason=response.status, spider=spider)
                if retryreq:
                    retryreq.meta['retry_times'] = retries
                    retryreq.meta['max_retry_times'] = self.max_retry_times
                    retryreq.meta['download_timeout'] = 30 * (2 ** retries)  # Exponential backoff
                    retryreq.dont_filter = True
                    return retryreq
        return response