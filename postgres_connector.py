from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from torch import Tensor
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class TextEmbedding(Base):
    __tablename__ = 'test_embeddings'  # change later
    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding = Column(Vector)
    content = Column(String)
    file_name = Column(String)
    sentence_number = Column(Integer)

    def __str__(self):
        return self.content + " " + str(self.id)

def get_psql_session(db_name: str):
    engine = create_engine(f'postgresql://postgres:postgres@localhost/{db_name}')
    Base.metadata.create_all(bind=engine)                                                                                                                                                                                                                     
    Session = sessionmaker(bind=engine, autoflush=False)
    return Session()

def upload_embeddings_into_db(session:Session, embeddings:Tensor, sentences:list[str], filename: str):
    for i, (embedding, content) in enumerate(zip(embeddings, sentences)):
        new_embedding = TextEmbedding(
            embedding=embedding, 
            content=content, 
            file_name=filename, 
            sentence_number=i+1
        )
        #print(f"new_embedding: {new_embedding}")
        session.add(new_embedding)
    session.commit()

# Finding the content from our database which is most similar to the query
def search_embeddings(query_embedding, session, limit=5):
    return session.query(TextEmbedding.id, TextEmbedding.sentence_number, TextEmbedding.content, TextEmbedding.file_name, 
        TextEmbedding.embedding.cosine_distance(query_embedding).label("distance") )\
        .order_by("distance").limit(limit).all()