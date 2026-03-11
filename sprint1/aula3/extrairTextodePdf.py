import fitz

doc = fitz.open(r"C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\aula3\pdfImagem.pdf")
page = doc[0]
text = page.get_text()

if text.strip():
    print("PDF possui texto digital")
    print(text)
else:
    print("PDF provavelmente é escaneado")

# FOCAR EM extrair texto que Não tem algo ou tem
# Texto OCR para esse texto