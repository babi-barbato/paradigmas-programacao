import pytesseract
from PIL import Image

def perform_ocr (image_path):
    # Abrir a Imagem
    image = Image.open(image_path)

    # Realizar OCR usando tesseract
    text = pytesseract.image_to_string(image, lang='por')
    return text

image_path = r"C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\aula3\imagem-exemplo.png"

print("Realizando OCR...")
ocr_text = perform_ocr(image_path)
print(ocr_text)
print()