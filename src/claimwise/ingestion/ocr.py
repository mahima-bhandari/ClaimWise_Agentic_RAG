
from PIL import Image
import pytesseract
import pymupdf


def extract_page_text_with_ocr(
    page: pymupdf.Page,
    language: str = "eng",
    dpi: int = 200,
) -> str:
    """
    Render one PDF page as an image and extract
    its text using Tesseract OCR.
    """

    # 1. Render the PDF page into a raster image.
    pixmap = page.get_pixmap(
        dpi=dpi,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    # 2. Convert the rendered image into a PIL Image.
    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    # 3. Recognise text in the image using Tesseract.
    try:
        extracted_text = pytesseract.image_to_string(
            image,
            lang=language,
            config="--psm 3",
        )

    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR executable was not found. "
            "Please check its installation and PATH."
        ) from error

    except pytesseract.TesseractError as error:
        raise RuntimeError(
            f"Tesseract OCR failed: {error}"
        ) from error

    return extracted_text.strip()
