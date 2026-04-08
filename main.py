import os
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from src.bot import start, handle_message

# Cargamos el fichero .env y guardamos las variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


def main():
    
    # Comprobamos que el token se haya cargado
    if not TELEGRAM_TOKEN:
        print("Error: No se ha encontrado el token de Telegram. Asegúrate de que el archivo .env contiene TELEGRAM_TOKEN.")
        return
    
    print ("Iniciando bot de telegram...")
    
    # Creamos la app del bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Añadimos los handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print ("Bot iniciado y en escucha. Presiona Ctrl+C para detenerlo.")
    
    app.run_polling()
    

if __name__ == "__main__":
    main()