import re
import uuid
from typing import Iterator, List

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.schema import Document
from pdf2image import convert_from_path
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_dict

from loguru import logger


class ImageWithNarrativeAndCaption:
    def __init__(self, id, image, page):
        self.id = id
        self.image = image
        self.caption = None
        self.narrative = None
        self.coords = None
        self.s3_path = None
        self.page = page

    def __repr__(self):
        return f"ImageWithNarrative(image={self.image}, caption={self.caption}, narrative={self.narrative})"


def convert_coords_to_tuple(coords):
    top_left, top_right, bottom_right, bottom_left = coords

    left, upper = top_left
    right, lower = bottom_right
    return (left, upper), (right, lower)


def extract_coords(i):
    if "coordinates" in i:
        coords = i["coordinates"]
    elif (
        "metadata" in i
        and "coordinates" in i["metadata"]
        and "points" in i["metadata"]["coordinates"]
    ):
        coords = i["metadata"]["coordinates"]["points"]
    else:
        raise Exception(f"Cannot find coordinates for {i}")
    return coords


def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)  # Removes any surrogate characters
    return text


class CustomPDFParser(BaseBlobParser):
    """Loads a PDF with pypdf and chunks at character level."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            yield from [
                Document(
                    page_content=clean_text(page.extract_text()),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]


def sort_elements(elements: list) -> list:
    return sorted(elements, key=lambda x: extract_coords(x)[0])


def get_page_text(elements: list, page_number: int) -> str:
    try:
        page_elements = [
            el for el in elements if el["metadata"].get("page_number", 0) == page_number
        ]
        sorted_elements = sort_elements(page_elements)

        column_threshold = max(extract_coords(el)[0][0] for el in sorted_elements) / 2

        left_column = [
            el for el in sorted_elements if extract_coords(el)[0][0] < column_threshold
        ]
        right_column = [
            el for el in sorted_elements if extract_coords(el)[0][0] >= column_threshold
        ]

        text_sequence = (
            " ".join([el["text"] for el in left_column])
            + " "
            + " ".join([el["text"] for el in right_column])
        )
        return text_sequence
    except Exception as e:
        logger.warning(
            f"Error {e} while processing page {page_number}, falling back to full page text"
        )

    text_sequence = " ".join(
        [
            el["text"]
            for el in elements
            if el["metadata"].get("page_number", 0) == page_number
        ]
    )
    return text_sequence


class CustomPDFLoader(BasePDFLoader):
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(self, file_path: str, multiplier=0.02) -> None:
        """Initialize with file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        self.parser = CustomPDFParser()
        self.multiplier = multiplier
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        try:
            if self.file_path.endswith(".pdf"):
                elements = partition_pdf(self.file_path, strategy="auto")
            else:
                elements = partition(filename=self.file_path)
            elements_dict = convert_to_dict(elements)
            pages = [el["metadata"].get("page_number", 0) for el in elements_dict]
            max_page = max(pages)
            min_page = min(pages)
            for page_number in range(min_page, max_page + 1):
                try:
                    yield Document(
                        page_content=clean_text(
                            get_page_text(elements_dict, page_number)
                        ),
                        metadata={"source": self.file_path, "page": page_number},
                    )
                except Exception as e:
                    import traceback

                    logger.info(f"Error {e} while processing page {page_number}")
                    logger.info("Traceback: ", traceback.format_exc())
                    continue

        except Exception as e:
            import traceback

            logger.info(f"Error {e} while processing file {self.file_path}")
            logger.info("Outer traceback: ", traceback.format_exc())
            return

    def cut_image_with_coords(self, image, coords):
        (left, upper), (right, lower) = convert_coords_to_tuple(coords)

        box = (
            left * (1 - self.multiplier),
            upper * (1 - self.multiplier),
            right * (1 + self.multiplier),
            lower * (1 + self.multiplier),
        )
        cropped_image = image.crop(box)
        return cropped_image

    def load_images(
        self,
        use_hi_res: bool = True,
    ) -> List[List[ImageWithNarrativeAndCaption]]:
        """Load given path as pages."""
        return list(self.lazy_load_images(use_hi_res))

    def lazy_load_images(
        self,
        use_hi_res: bool = True,
    ) -> Iterator[List[ImageWithNarrativeAndCaption]]:
        """ """

        logger.info(f"Partitioning pdf from {self.file_path}")

        elements = partition_pdf(
            self.file_path, strategy="auto"  # "hi_res" if use_hi_res else "auto"
        )

        logger.info(f"Creating images from {self.file_path}")

        images_of_pages = self.create_images_of_pages()

        pages_content = {}
        for i in convert_to_dict(elements):
            if i["type"] in ["PageBreak"]:
                continue  # skipping non-informative elements

            try:
                page_number = i["metadata"]["page_number"] - 1
            except Exception as e:
                logger.info(f"Error {e} while processing element {i}")
                continue

            if page_number not in pages_content:
                pages_content[page_number] = []
            pages_content[page_number].append(i)

        for page_number, page_image in enumerate(images_of_pages):
            images_on_page = []

            if page_number in pages_content:
                page_content = pages_content[page_number]

                for i in page_content:
                    try:
                        if i["type"] == "FigureCaption":
                            image = self.content_matching(
                                i, page_content, page_image, page_number
                            )
                            images_on_page.append(image)
                    except Exception as e:
                        import traceback

                        logger.info(traceback.format_exc())
                        logger.info(f"Error while processing image: {e}")

            yield images_on_page

    def create_images_of_pages(self):
        images_of_pages = convert_from_path(self.file_path, dpi=200, thread_count=4)
        return images_of_pages

    def content_matching(self, i, page_content, page_image, page_number):
        coords = extract_coords(i)
        cropped_image = self.cut_image_with_coords(page_image, coords)
        image_id = uuid.uuid4().hex
        image = ImageWithNarrativeAndCaption(image_id, cropped_image, page_number)
        image.caption = i["text"]
        image.coords = coords
        (left1, upper1), (right1, lower1) = convert_coords_to_tuple(coords)
        min_distance_y = float("inf")
        nearest_narrative_text = None
        for j in page_content:
            if j["type"] == "NarrativeText":
                coords2 = extract_coords(j)
                (left2, upper2), (right2, lower2) = convert_coords_to_tuple(coords2)

                distance_y = upper2 - lower1

                if 0 < distance_y < min_distance_y:
                    min_distance_y = distance_y
                    nearest_narrative_text = j["text"]
        image.narrative = nearest_narrative_text
        return image
