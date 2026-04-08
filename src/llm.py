from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Configuraciones
RUTA_REGLAS = "data/reglamento.pdf"
DIRECTORIO_DB = "chorma_db"

# Inicializamos el modelo de lenguaje de Ollama
llm = OllamaLLM(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def inicializar_bd():
    """Carga el PDF, lo divide en fragmentos y lo guarda en ChromaDB si no existe."""
    
    if os.path.exists(DIRECTORIO_DB):
        print("Cargando BD vectorial existente...")
        return Chroma(persist_directory=DIRECTORIO_DB, embedding_function=embeddings)
    

    print("Creando nueva BD vectorial (puede tardar un poco)...")
    loader = PyPDFLoader(RUTA_REGLAS)
    documentos = loader.load()
    
    # Dividimos el texto en fragmentos para mejorar la búsqueda
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos)
    
    # Guardamos los fragmentos en ChromaDB
    vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DIRECTORIO_DB)
    

# Cargamos la BD al iniciar el módulo
vectorstore = inicializar_bd()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Promp base para que la IA sea un experto en la materia
system_prompt = (
    """
    Eres un experto en reglamento de rugby. Lee cuidadosamente los textos proporcionados y contesta la siguiente pregunta basándote exclusivamente en su contenido.
    Si hay información insuficiente para responder, indícalo explícitamente.
    Si encuentras discrepancias o inconsistencias entre las fuentes, menciónalas y explica brevemente.

    Instrucciones para tu respuesta:
    1. Usa la información de los textos anteriores para responder.
    2. Ofrece la respuesta de forma clara y concisa.
    3. Incluye, si es posible, referencias o citas a la(s) fuente(s) que justifiquen tus afirmaciones.
    4. Si la información no está en los textos o no es concluyente, indícalo.
    5. Si encuentras discrepancias, menciónalas explícitamente.

    Ahora, por favor, elabora tu respuesta en base a la pregunta:
    """
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
                Eres un experto en reglamento de rugby. Lee cuidadosamente los textos proporcionados y contesta la siguiente pregunta basándote exclusivamente en su contenido.
                Si hay información insuficiente para responder, indícalo explícitamente.
                Si encuentras discrepancias o inconsistencias entre las fuentes, menciónalas y explica brevemente.
                
                Contexto recuperado del reglamento:
                {context}

                Instrucciones para tu respuesta:
                1. Usa la información de los textos anteriores para responder.
                2. Ofrece la respuesta de forma clara y concisa.
                3. Incluye, si es posible, referencias o citas a la(s) fuente(s) que justifiquen tus afirmaciones.
                4. Si la información no está en los textos o no es concluyente, indícalo.
                5. Si encuentras discrepancias, menciónalas explícitamente.

                Ahora, por favor, elabora tu respuesta en base a la pregunta:
                """),
    ("human", "{input}"),
])

# Creamos la cadena RAG
question_answering_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

def generar_respuesta(pregunta: str) -> str:
    """
        Recibe la pregunta del usuario, procesa la información,
        consulta la BD y genera una respuesta usando un sistema RAG.
    """
    
    try:
        respuesta = rag_chain.invoke({"input": pregunta})
        return respuesta["answer"]
    
    except Exception as e:
        return f"Lo siento, ha ocurrido un error al generar la respuesta: {str(e)}"