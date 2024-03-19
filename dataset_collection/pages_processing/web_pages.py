import multiprocessing
import re
import uuid
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import requests
import unstructured.documents.html as HTT
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from selenium import webdriver
from timeout_decorator import timeout
from unstructured.cleaners.core import (
    clean_bullets,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    clean_ordered_bullets,
)
from unstructured.partition.html import partition_html
from loguru import logger


def clean_unstructured_text(el):
    text = str(el)
    text = clean_non_ascii_chars(text)
    text = clean_extra_whitespace(text)
    text = clean_bullets(text)
    try:
        text = clean_ordered_bullets(text)
    except:
        pass
    return text


def clean_text_new(text):
    # Parse the HTML with Beautiful Soup
    soup = BeautifulSoup(text, "html.parser")

    # replace <br/> tags with \n
    for br in soup.find_all("br"):
        br.replace_with("")

    for class_ in ["NavigationFooter", "GlobalNavigation", "global-header"]:
        element_to_remove = soup.find(class_=class_)
        if element_to_remove:
            element_to_remove.extract()

    # Remove unnecessary elements
    for script in soup(["script", "style"]):
        script.decompose()  # removes the tag

    for tag in soup.select('[aria-hidden="true"]'):
        tag.decompose()

    cleaned_html = soup.prettify()
    elements = partition_html(text=cleaned_html)

    clean_text = ""
    for el in elements:
        if type(el) == HTT.HTMLListItem:
            clean_text += "- " + clean_unstructured_text(el)
            clean_text += "\n"
        else:
            clean_text += clean_unstructured_text(el)
            clean_text += "\n"
    return clean_text


def clean_text(text):
    # Remove titles and navigation elements
    text = re.sub(r"Skip to Main Content", "", text)
    text = re.sub(r"(top of|bottom of) page", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Use tab to navigate through the menu items.", "", text)
    text = re.sub(r"Open site navigation", "", text)
    text = re.sub(r"Log In", "", text)
    text = re.sub(r"- Home", "", text)
    text = re.sub(r"- More", "", text)

    # Remove empty lines
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    text = "\n".join(lines)
    return text.strip()


def process_text_item(doc, identifiers: Dict[str, str], is_preprocessed=False):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", " "],
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
    )

    try:
        # if not is_preprocessed:
        #     content = clean_text(html2text(doc["content"]))
        # else:
        #     content = clean_text(doc["content"])
        content = doc["clean_text"]
        url = doc["url"]
        title = doc["title"]
        content = f"""URL: {url}\nTITLE: {title}\n{content}"""
        # print(f"url: {url}, title: {title}, content: {content}")

        meta_data = {"source": url, "title": doc["title"]}
        meta_data.update(identifiers)

        doc = Document(page_content=content, metadata=meta_data)
        docs = text_splitter.split_documents([doc])
        is_success = True
    except:
        logger.debug(f"Fail to process text for {doc['url']}")
        docs = []
        is_success = False

    return docs, is_success


def text_preprocess(docs: List, identifiers: Dict[str, str], is_preprocessed=False):
    pool = multiprocessing.Pool()

    results = [
        pool.apply_async(process_text_item, args=(doc, identifiers, is_preprocessed))
        for doc in docs
    ]

    new_docs = []
    statuses = []
    for result in results:
        docs, is_success = result.get()
        new_docs.append(docs)
        statuses.append(is_success)

    pool.close()
    pool.join()
    return new_docs, statuses


def get_parent_url(url):
    parsed_url = urlparse(url)
    parent_url = parsed_url.scheme + "://" + parsed_url.netloc
    return parent_url


def format_image_text(caption, url, above_text, below_text):
    img_name = url.split("/")[-1]
    output = f"""Caption of {img_name} is: {caption}
Description: {above_text}
{below_text}
    """
    return output


def process_image_item(doc, identifiers: Dict[str, str]):
    image_docs = []
    try:
        pattern = r"!\[([^\]]*)\]\(([^)]*)\)"
        content = doc["clean_image_text"]
        matches = re.findall(pattern, content)

        for match in matches:
            try:
                url = match[1]
                if ".jpg" not in url and ".png" not in url and ".jpeg" not in url:
                    continue
                caption = match[0]

                url_with_caption = f"[{caption}]({url})"
                above_text = "\n".join(
                    content[: content.find(url_with_caption) - 1].split("\n\n")[-2:]
                )
                below_text = "\n".join(
                    content[
                        content.find(url_with_caption) + len(url_with_caption) :
                    ].split("\n\n")[:2]
                )

                if len(" ".join([caption, above_text, below_text]).split()) < 1:
                    continue

                image_id = uuid.uuid4().hex
                if not url.startswith("https") and not url.startswith("http"):
                    parent_url = get_parent_url(doc["url"])
                    url = urljoin(parent_url, url)

                metadata = {
                    "image_id": image_id,
                    "s3_path": url,
                    "source": doc["url"],
                    "url": doc["url"],
                    "caption": caption,
                    "narrative": f"{above_text}, {below_text}",
                    "page": 0,
                }
                metadata.update(identifiers)
                document = Document(
                    page_content=format_image_text(
                        caption, url, above_text, below_text
                    ),
                    metadata=metadata,
                )
                image_docs.append(document)
            except:
                logger.debug(f"Error with the url: {match[1]}")
        is_success = True
    except:
        is_success = False
    return image_docs, is_success


def images_preprocess(docs: List, identifiers: Dict[str, str]):
    pool = multiprocessing.Pool()
    results = [
        pool.apply_async(process_image_item, args=(doc, identifiers)) for doc in docs
    ]

    new_docs = []
    statuses = []
    for result in results:
        docs, is_success = result.get()
        new_docs.append(docs)
        statuses.append(is_success)

    pool.close()
    pool.join()
    return new_docs, statuses


@timeout(20)  # 20s
def get_final_redirected_url(url):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument(
            "--headless"
        )  # Run Chrome in headless mode (without a visible browser window)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)

        timeout_seconds = 10
        driver.implicitly_wait(timeout_seconds)

        driver.get(url)

        redirected_url = driver.current_url

        driver.quit()
        return redirected_url
    except Exception as e:
        logger.debug(f"Error for redirecting: {e}")
        return url


def web_crawler(link: str, max_pages: int, max_chars: int, identifiers: Dict[str, str]):
    if link[:4] != "http" and link[:5] != "https":
        link = "https://" + link
    # check redirecting
    redirected_url = get_final_redirected_url(url=link)
    logger.debug(f"REDIRECTING from {link} to {redirected_url}")

    crawled_docs = requests.post(
        Config.CRAWLING_SERVICE_URL,
        json={
            "url": redirected_url,
            "max_pages": max_pages,
            "max_chars": max_chars,
            "identifiers": identifiers,
        },
    ).json()
    if crawled_docs["success"] == False:
        return []

    final_crawled_docs = []
    crawled_urls = []
    for doc in crawled_docs["data"]:
        url = doc["url"].replace("www.", "").rstrip("/")
        if url not in crawled_urls:
            crawled_urls.append(url)
            final_crawled_docs.append(doc)

    return final_crawled_docs


def process_crawled_text_docs(
    crawled_docs: List,
    identifiers: Dict[str, str],
):
    all_docs, all_statuses = text_preprocess(docs=crawled_docs, identifiers=identifiers)
    logger.info("Finish web text preprocess")

    merged_docs = []
    doc_mapping = {}
    for i, docs in enumerate(all_docs):
        if all_statuses[i]:
            for idx in range(len(merged_docs), len(merged_docs) + len(docs)):
                doc_mapping[idx] = i
            merged_docs.extend(docs)

    return merged_docs, doc_mapping


def process_crawled_image_docs(
    crawled_docs: List,
    identifiers: Dict[str, str],
):
    all_docs, all_statuses = images_preprocess(
        docs=crawled_docs, identifiers=identifiers
    )
    logger.info("Finish web text preprocess")

    merged_docs = []
    doc_mapping = {}
    img_urls = []
    for i, docs in enumerate(all_docs):
        if all_statuses[i]:
            new_image_docs = []
            for doc in docs:
                if doc.metadata["s3_path"] not in img_urls:
                    img_urls.append(doc.metadata["s3_path"])
                    new_image_docs.append(doc)

            for idx in range(len(merged_docs), len(merged_docs) + len(new_image_docs)):
                doc_mapping[idx] = i

            merged_docs.extend(new_image_docs)

    return merged_docs, doc_mapping
