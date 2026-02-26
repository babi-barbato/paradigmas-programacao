import pytesseract
from PIL import Image
from pdf2image import convert_from_path

poppler_path = r"C:\poppler-26.02.0"

def extrect_text_from_pdf(pdf_path):
    # Converter o PDF em uma lista de imagens
    imagens = convert_from_path(pdf_path, poppler_path=poppler_path)

    texto_extraido = ""

    # Iterar sobre cada imagem e extrair usando OCR 
    for i, imagem in enumerate(imagens):
        texto_imagem =pytesseract.image_to_string(imagem, lang='por')
        texto_extraido += f"Texto da pagina {i+1};\n{texto_imagem}\n\n"

    return texto_extraido

# Caminho da Imagem
image_path = "C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\aula3\pdfImagem.pdf"

# Realizar OCR na imagem
ocr_text = extrect_text_from_pdf(image_path)
print("Texto extraído do PDF:")
print(ocr_text)