from django.core.management.base import BaseCommand
from chatbotHRapp.models import HRDocument
from chatbotHRapp.rag import extract_text_from_pdf, chunk_text_content, store_in_pinecone

class Command(BaseCommand):
    help = "Ingest and embed PDF files into Pinecone."

    def handle(self, *args, **kwargs):  # Add *args and **kwargs
        # Query all unprocessed HRDocument rows
        docs = HRDocument.objects.filter(processed=False)

        for doc in docs:
            pdf_path = doc.file.path
            text = extract_text_from_pdf(pdf_path)

            # 1. Chunk the text
            chunks = chunk_text_content(text, chunk_size=1000, chunk_overlap=100)

            # 2. Store chunks in Pinecone
            store_in_pinecone(doc.id, chunks)

            # Mark as processed
            doc.processed = True
            doc.save()

        self.stdout.write(self.style.SUCCESS("Ingestion complete!"))