import traceback

import unstructured.documents.html as HTT
from bs4 import BeautifulSoup
from loguru import logger
from unstructured.cleaners.core import (
    clean_bullets,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    clean_ordered_bullets,
)
from unstructured.partition.html import partition_html

from scraper_processor.services.html2text import html2text_with_images


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

    for class_ in [
        "NavigationFooter",
        "GlobalNavigation",
        "global-header",
        "site-footer",
        "site-header",
        "twitter-feed",
        "navbar",
        "footer",
        "navbar",
    ]:
        element_to_remove = soup.find(class_=class_)
        if element_to_remove:
            element_to_remove.extract()

    for header_id in ["navbar"]:
        element_to_remove = soup.find("header", id=header_id)
        if element_to_remove:
            element_to_remove.extract()

    # Remove the identified footer elements from the document
    footer_elements = soup.find_all("footer")
    for footer in footer_elements:
        footer.decompose()

    # Remove unnecessary elements
    for script in soup(["script", "style"]):
        script.decompose()  # removes the tag

    # for tag in soup.select('[aria-hidden="true"]'):
    #     tag.decompose()

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


def webpage_processor(raw_text):
    try:
        clean_text = clean_text_new(raw_text)
    except:
        logger.debug(
            f"Error doing webpage processing text, error {traceback.format_exc()}"
        )
        clean_text = None

    try:
        clean_image_text = html2text_with_images(raw_text)
    except:
        logger.debug(
            f"Error doing webpage processing image, error {traceback.format_exc()}"
        )
        clean_image_text = None

    return clean_text, clean_image_text
