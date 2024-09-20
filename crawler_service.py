from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
# from spiders.general_spider import GeneralSpider
from scrape_it import GeneralSpider

class CrawlerService:
    def __init__(self):
        self.process = CrawlerProcess(get_project_settings())

    def run_crawler(self, start_url):
        self.process.crawl(GeneralSpider, start_url=start_url)
        self.process.start()

if __name__ == "__main__":
    service = CrawlerService()
    website_url = input("Enter the website URL to crawl: ")
    service.run_crawler(website_url)