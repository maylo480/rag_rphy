from reportlab.pdfgen import canvas

def save_sentence_to_pdf(sentences:list[str], filename):
    y = 750
    c = canvas.Canvas(f"test_pdfs/{filename}.pdf")
    for sentence in sentences:
        c.drawString(100, y, sentence)
        y -= 20
    c.save()

pianka_sentences = ["Pianka is my dog", "Pianka is a JapaneseSpitz", "Pianka likes rybka", "My dog likes to play with paddle ball", "My dog will be 2 years old on 16th of February 2026"]
ala_sentences = ["Ala is my girlfriend", "Ala is an ETL developer", "Ala works in mBank", "My girlfriend has blonde hair", "My girlfriend will be 24 years old on 16th of May 2026"]
volv_sentences = ["Volvo V70 is my car", "My car is a wagon", "Volvo V70 has a 2,5litre 20V engine", "My car's fuel consumption is about 10liters per 100km", "My Volvo V70 was produces in 1997"]
save_sentence_to_pdf(pianka_sentences, "pianka")
save_sentence_to_pdf(ala_sentences, "ala")
save_sentence_to_pdf(volv_sentences, "volv")