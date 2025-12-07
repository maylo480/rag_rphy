from sentence_transformers import SentenceTransformer
import nltk
import pdfplumber
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String
from c_03_database_connect_embeddings import TextEmbedding
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pgvector.sqlalchemy import Vector
import gc
from ollama import chat
from utils import get_surrounding_sentences, search_embeddings, get_filtered_matches

RPHY_PDF = "pdfs/CM-SP-R-PHY-I20-250402.pdf"

def convert_pdf_to_sentences(filename=RPHY_PDF):
    text_chunks = []
    print(f"Opening {filename}")
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            text_chunks.append(page.extract_text() or "")
    full_text = "\n".join(text_chunks)
    # basic sentence split; for production use nltk/spacy
    sentences = [s.strip() for s in full_text.split(".") if s.strip()]
    print(f"{len(sentences)} generated from pdf file")
    print(f"Saving to a txt file")
    with open(file="aa.txt", mode='w') as f:
        for sentence in sentences:
            f.write(sentence)
    return sentences

def get_psql_session():
    engine = create_engine('postgresql://postgres:postgres@localhost/text_embeddings')
    Base = declarative_base()
    Base.metadata.create_all(engine)

    # Create a session                                                                                                                                                                                                                      
    Session = sessionmaker(bind=engine)
    return Session()

def populate_vector_database():
    nltk.download("punkt")
    nltk.download("punkt_tab")

    session = get_psql_session()
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    filename = "aa.txt"
    print(f"Generating embeddings for file {filename}")
        
    try:
        with open(file=filename, mode="r") as f:
            full_text = f.read()
        sentences = [s for s in full_text.split("\n")]
        print(f"Extracted {len(sentences)} sentences from a txt file")
        # embeddings = embed(model="custom_deepseek", input=sentences)["embeddings"]
        embeddings = model.encode(sentences, show_progress_bar=True)
        
        for i, (embedding, content) in enumerate(zip(embeddings, sentences)):
            new_embedding = TextEmbedding(embedding=embedding, content=content, file_name=filename, sentence_number=i+1)
            # print(f"new_embedding: {new_embedding}")
            session.add(new_embedding)
        session.commit()

        print("Succesfully generated embeddings for: {}".format(filename))

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
    return

def search_by_query(session, query, num_matches=5, group_window_size=5):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    query_embedding = model.encode(query)
    del model
    gc.collect()

    search_results = search_embeddings(query_embedding, session=session, limit=num_matches * (2*group_window_size + 1) )
    filtered_matches = get_filtered_matches(search_results)

    entry_ids = [i[0] for i in filtered_matches]
    file_names = [i[3] for i in filtered_matches]

    return get_surrounding_sentences(entry_ids=entry_ids, file_names=file_names, group_window_size=group_window_size, session=session)

def provide_context_for_query(query: str, session: Session):
    context = search_by_query(
        session=session,
        query=query
    )
    print(f"Context: {context}")

    prompt = f"<|content_start>{context} \
    <|content_end> {query}"

    print(f"Prompt: {prompt}")
    return prompt

def main():
    query = "What is RPD?"
    psql_session = get_psql_session()

    prompt = provide_context_for_query(query=query, session=psql_session)

    response = chat(model='hf.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    print(response.message.content)

if __name__ == "__main__":
    main()