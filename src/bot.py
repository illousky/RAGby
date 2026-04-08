from telegram import Update
from telegram.ext import ContextTypes
from src.llm import generar_respuesta


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    # Respuesta al iniciar la conversación
    mensaje_bienvenida = (
        "¡Hola! Soy el asistente virtual del equipo de Rugby de la UAM 🏉.\n\n"
        "¡Pregúntame lo que quieras sobre reglamento de rugby!"
    )
    
    await update.message.reply_text(mensaje_bienvenida)
    
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Recibe el mensaje y se lo envia a la IA para generar una respuesta."""
    
    mensaje_usuario = update.message.text
    
    # Enviamos una confirmación de que se ha recibido el mensaje y se está procesando
    mensaje_espera = await update.message.reply_text("Estoy consultando el reglamento...")
    
    # Llamamos a la IA mediante Ollama para generar la respuesta
    respuesta_ia = generar_respuesta(mensaje_usuario)
    
    # Editamos el mensaje de espera con la respuesta obtenida de la IA
    await mensaje_espera.edit_text(respuesta_ia)