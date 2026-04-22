from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import os

# Configuración
RUTA_REGLAS = "data/reglamento.pdf"
DIRECTORIO_DB = "chroma_db"

# LLM con TEMPERATURA 0 para máxima precisión
llm = OllamaLLM(model="llama3", temperature=0.3)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def inicializar_bd():
    if os.path.exists(DIRECTORIO_DB):
        return Chroma(persist_directory=DIRECTORIO_DB, embedding_function=embeddings)
    loader = PyPDFLoader(RUTA_REGLAS)
    fragmentos = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300).split_documents(loader.load())
    return Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DIRECTORIO_DB)

vectorstore = inicializar_bd()

def generar_respuesta(pregunta: str) -> str:
    try:
        # 1. TRADUCCIÓN TÉCNICA
        prompt_re = f"""Convierte esta consulta de usuario en términos técnicos de World Rugby (pelota, try, scrum, knock-on, penal). 
        Consulta: {pregunta}
        Respuesta (SOLO LA CONSULTA TÉCNICA):"""
        p_oficial = llm.invoke(prompt_re).strip()
        
        # 2. RECUPERACIÓN Y RERANKING
        docs = vectorstore.similarity_search(p_oficial, k=20)
        scores = reranker_model.predict([[p_oficial, d.page_content] for d in docs])
        docs_puntuados = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        
        # Solo usamos fragmentos con una puntuación mínima de relevancia
        mejores_textos = [d.page_content for score, d in docs_puntuados if score > 0.05][:4]
        
        if not mejores_textos:
            mejores_textos = [d.page_content for d in docs[:2]]  # fallback a los primeros 2 si no hay buenos matches

        contexto_unido = "\n\n".join(mejores_textos)

        # 3. PROMPT DE PRODUCCIÓN (EL "JUEZ")
        prompt_final = f"""
        Eres un veterano de un club de rugby español. Tu estilo es castizo, directo y experto. 
        No eres un robot, eres un compañero que explica las reglas a los que están empezando para que el sábado no nos inflen a golpes de castigo.

        ### NORMAS DE ESTILO:
        - Usa "avante" o "knock-on". No uses "knock-forward".
        - Usa "melé" o "scrum".
        - Sé directo. Si es melé, di que es melé. No des rodeos motivadores.
        - Si no lo sabes por el contexto, no te lo inventes.

        ### EJEMPLO DE RESPUESTA IDEAL:
        "Mira, eso es un avante. Se te ha caído la pelota hacia adelante en el contacto y eso es infracción. El árbitro va a pitar melé para el otro equipo. Ellos introducen la pelota en el punto donde se te cayó y a jugar."

        ### CONTEXTO REGLAMENTARIO:
        {contexto_unido}

        ### PREGUNTA DEL JUGADOR:
        {p_oficial}

        ### EXPLICACIÓN DEL VETERANO (responde solo un párrafo natural):
        """
        
        return llm.invoke(prompt_final)
        
    except Exception as e:
        return f"Error técnico en el motor: {str(e)}"