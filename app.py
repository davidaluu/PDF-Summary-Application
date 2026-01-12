from flask import Flask, render_template, request, send_file
from openai import OpenAI
import io
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
from reportlab.pdfbase.pdfmetrics import stringWidth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = Flask(__name__)


# ----- PDF Handling -----
def pdf_to_text(file):
    reader = PyPDF2.PdfReader(file.stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ----- Chunking -----
def chunk_text(text, max_tokens=2000):
    chunk_size = max_tokens * 4  # approx 1 token ~ 4 chars
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ----- OpenAI Summarization -----
def summarize_chunk(chunk):
    prompt = f"You are a document summarization assistant. You are receiving a portion of a larger document, and your task is to identify and extract the key points, main ideas, important information, critical data, statistics, findings, and conclusions from this section, then create a clear, concise summary that captures the essential content using professional language while maintaining the original meaning and focusing only on the content in this section, outputting only the summary text in paragraph form without any preamble, meta-commentary, or explanations. Word limit is 100 words\n{chunk}"
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def summarize_text(text):
    chunks = chunk_text(text)
    with ThreadPoolExecutor(max_workers=5) as executor:
        summaries = list(executor.map(summarize_chunk, chunks))
    return "\n\n".join(summaries)

# ----- PDF Generation with centered footer and page numbers -----
def create_summary_pdf(summary):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=LETTER)
    width, height = LETTER
    x = inch
    y = height - inch
    line_height = 14
    footer_text = "Copyright David Aluu 2026. All rights reserved"
    page_number = 1

    def draw_footer(page_num):
        text = f"{footer_text} | Page {page_num}"
        text_width = stringWidth(text, "Helvetica", 10)
        c.setFont("Helvetica", 10)
        c.drawString((width - text_width) / 2, 0.5 * inch, text)

    for line in summary.split("\n"):
        wrapped_lines = textwrap.wrap(line, width=90)
        for segment in wrapped_lines:
            if y < inch:
                draw_footer(page_number)
                c.showPage()
                page_number += 1
                y = height - inch
            c.drawString(x, y, segment)
            y -= line_height

    # Draw footer on last page
    draw_footer(page_number)
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# ----- Routes -----
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('document')
        if file:
            text = pdf_to_text(file)
            summary = summarize_text(text)
            pdf_file = create_summary_pdf(summary)
            return send_file(
                pdf_file,
                as_attachment=True,
                download_name="summary.pdf",
                mimetype="application/pdf"
            )
    return render_template('index.html')

# ----- Run App -----
if __name__ == '__main__':
    app.run(debug=True)


