#from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus.tables import Table
from reportlab.lib import colors
import base64
from PIL import Image
#from io import StringIO
from io import BytesIO
from reportlab.lib.utils import ImageReader


class CreatePDF:
    @staticmethod
    def create(base64image, neuro, extra, off, ver, gew):
        img_buffer = BytesIO()
        imgdata = base64.b64decode(base64image)
        img_buffer.write(imgdata)
        img_buffer.seek(0)
        img = ImageReader(img_buffer)
        (sx, sy) = img.getSize()

        # Create PDF as Buffer to circumvent saving it to file.
        # Instead we want to return it in base64 format
        pdf_buffer = BytesIO()
        pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=A4)

        pdf_canvas.setFillColorRGB(87/256, 104/256, 128/256)
        pdf_canvas.drawImage(img, 30, sy*0.34, width=sx*0.26, height=sy * 0.26)
        pdf_canvas.setLineWidth(.3)
        pdf_canvas.setFont('Helvetica-Bold', 24)
        pdf_canvas.drawString(30, 800, 'Ergebnis Deines Persönlichkeitstests')
        pdf_canvas.setFont('Helvetica', 12)
        pdf_canvas.drawString(30, 770, 'Hier kannst Du nun Dein eigenes Persönlichkeitsprofil als Netzdiagramm erkennen und siehst')
        pdf_canvas.drawString(30, 755, 'wie stark die fünf Merkmale bei Dir ausgeprägt sind.')

        data = [["Offenheit", "Extraversion", "Verträglichkeit", "Gewissenhaftigkeit", "Neurotizismus"], [str(off) + ' %', str(extra) + ' %', str(ver) + ' %', str(gew) + ' %', str(neuro) + ' %']]

        table = Table(data, colWidths=100)
        table.setStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, "#57687F"),
                       ("TEXTCOLOR", (0, 0), (-1, -1), "#57687F")])

        table.wrapOn(pdf_canvas, 1000, 400)
        table.drawOn(pdf_canvas, 30,350)

        pdf_canvas.setFont('Helvetica-Bold', 10)
        pdf_canvas.drawString(30, 320, '• Offenheit ')
        pdf_canvas.setFont('Helvetica', 10)
        pdf_canvas.drawString(80, 320, '  zeigt, wie stark Dich neue Erfahrungen oder Eindrücke interessieren. Hier geht es um Deine Einschätzung')
        pdf_canvas.drawString(30, 308, '  zu Fantasie, Ästhetik, Emotionalität, Neugier, Intellektualismus, Liberalismus.')
        pdf_canvas.drawString(30, 296, '  Je höher hier Dein Prozentwert, desto offener für Neues bist Du.')

        pdf_canvas.setFont('Helvetica-Bold', 10)
        pdf_canvas.drawString(30, 276, '• Extraversion ')
        pdf_canvas.setFont('Helvetica', 10)
        pdf_canvas.drawString(96, 276, '  beschreibt Deine Aufgeschlossenheit anderen Menschen gegenüber. Daher fragt der Test auch nach')
        pdf_canvas.drawString(30, 262, '  Freundlichkeit, Geselligkeit, Durchsetzungsfähigkeit, Aktivität, Abenteuerlust und Heiterkeit.')
        pdf_canvas.drawString(30, 250, '  Je höher Dein Prozentwert, desto aufgeschlossener bist Du.')

        pdf_canvas.setFont('Helvetica-Bold', 10)
        pdf_canvas.drawString(30, 230, '• Verträglichkeit ')
        pdf_canvas.setFont('Helvetica', 10)
        pdf_canvas.drawString(106, 230, '  beschäftigt sich mit der Frage, wie Du mit anderen Menschen hinsichtlich Vertrauen, Ehrlichkeit,')
        pdf_canvas.drawString(30, 218, '  Altruismus, Entgegenkommen, Bescheidenheit und Mitgefühl umgehst.')
        pdf_canvas.drawString(30, 206, '  Je höher hier Dein Prozentwert, desto „verträglicher“ bist Du im Umgang mit anderen.')

        pdf_canvas.setFont('Helvetica-Bold', 10)
        pdf_canvas.drawString(30, 186, '• Gewissenhaftigkeit ')
        pdf_canvas.setFont('Helvetica', 10)
        pdf_canvas.drawString(125, 186, '  beschreibt wie sorgfältig Du Deinen Aufgaben nachgehst. Je höher hier Dein Wert ist, desto')
        pdf_canvas.drawString(30, 174, '  ausgeprägter sind bei Dir Kompetenz, Ordnungsliebe, Pflichtbewusstsein, Leistungsstreben, Selbstdisziplin')
        pdf_canvas.drawString(30, 162, '  und Sorgfalt.')

        pdf_canvas.setFont('Helvetica-Bold', 10)
        pdf_canvas.drawString(30, 142, '• Neurotizismus ')
        pdf_canvas.setFont('Helvetica', 10)
        pdf_canvas.drawString(102, 142, '  zeigt Dir, wie verletzlich bzw. emotional instabil Du bist. Je höher der Wert, desto ausgeprägter')
        pdf_canvas.drawString(30, 130, '  sind bei Dir Ängstlichkeit, Reizbarkeit, Pessimismus, Befangenheit, Impulsivität und Verletzlichkeit.')


        pdf_canvas.setFillColorRGB(229 / 256, 236 / 256, 246 / 256)
        pdf_canvas.rect(20, 10, 555, 105, fill=1)
        pdf_canvas.setFillColorRGB(87/256, 104/256, 128/256)

        pdf_canvas.setFont('Helvetica', 9)
        pdf_canvas.drawString(30, 100, 'Du merkst, die Zusammenfassung der Merkmale und ihrer Bedeutungen ist nicht leicht – die Lesart des Netzdiagrammes macht')
        pdf_canvas.drawString(30, 88, 'das nicht leichter ;-)')

        pdf_canvas.drawString(30, 76, 'Daher: Bei dem Netzdiagramm gibt es keine guten oder schlechten Ausprägungen. Gerade die unterschiedlichen Ausprägungen')
        pdf_canvas.drawString(30, 64, 'der Persönlichkeitsmerkmale machen uns einzigartig und haben alle ihre Berechtigung.')

        pdf_canvas.drawString(30, 40, 'Schau Dir Dein Ergebnis ganz unvoreingenommen an.')

        pdf_canvas.setFont('Helvetica-Bold', 9)
        pdf_canvas.drawString(30, 20, 'Vielleicht findest Du etwas über Dich heraus, vielleicht hat sich Cleo aber auch geirrt – Das entscheidest am Ende Du!')
        pdf_canvas.drawString(30, 8, '')

        pdf_canvas.save()
        pdf_file = pdf_buffer.getvalue()
        pdf_buffer.close()
        img_buffer.close()

        return base64.b64encode(pdf_file).decode()



# if __name__ == '__main__':
#     with open("../../images/fig0.png", "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     CreatePDF.create(encoded_string, 80, 50, 60, 20, 10)
