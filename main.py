import logging
import os
import io
from PIL import Image
import re

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.constants import ChatAction, ParseMode

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig

from dotenv import load_dotenv

# --- Database Utilities ---
import database # Import your new database module

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Logging Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Gemini AI Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"🚨 Error configuring Gemini API: {e}")

# Safety settings for Gemini
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Primary model for image-related interactions
PRIMARY_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
# Fallback text model for general chat
FALLBACK_TEXT_MODEL_NAME = "gemini-1.5-flash"

# System instruction for the FALLBACK Gemini model
FALLBACK_SYSTEM_INSTRUCTION = (
    "You are a helpful, creative, and multilingual AI assistant. "
    "Respond to the user's message with text only. " 
    "Always strive to respond in the same language the user is using for their message. "
    "Be conversational and engaging."
)

gemini_primary_model = None
gemini_fallback_text_model = None

try:
    gemini_primary_model = genai.GenerativeModel(
        PRIMARY_MODEL_NAME,
        safety_settings=SAFETY_SETTINGS
    )
    logger.info(f"✅ Successfully initialized primary multimodal model: {PRIMARY_MODEL_NAME}")
except Exception as e:
    logger.error(f"🚨 Error initializing primary multimodal model ({PRIMARY_MODEL_NAME}): {e}. It will be unavailable.")

try:
    gemini_fallback_text_model = genai.GenerativeModel(
        FALLBACK_TEXT_MODEL_NAME,
        safety_settings=SAFETY_SETTINGS,
        system_instruction=FALLBACK_SYSTEM_INSTRUCTION 
    )
    logger.info(f"✅ Successfully initialized fallback text model: {FALLBACK_TEXT_MODEL_NAME}")
except Exception as e_fallback:
    logger.error(f"🚨 Error initializing fallback text model ({FALLBACK_TEXT_MODEL_NAME}): {e_fallback}. Text chat will be impaired if primary also failed.")


# --- User Context Store (In-memory for current turn state, DB for primary model's long-term history) ---
user_context = {} # chat_id -> {"last_image_bytes": BytesIO, "last_mode": str, 
                  #             "fallback_chat_session": ChatSession, "active_model_name": str}
                  # "primary_model_history" is now managed via the database.py module

# Keywords for the bot to decide if an image generation/interaction is likely intended for the primary model
IMAGE_GENERATION_KEYWORDS = [
    "draw", "generate image", "create a picture", "make an image", "show me an image of",
    "picture of", "image of", "photo of", "illustration of", "sketch of"
]


# --- Helper Functions ---
def store_last_image(chat_id, image_bytes):
    """Stores the last processed image in memory for immediate follow-up actions."""
    str_chat_id = str(chat_id)
    if str_chat_id not in user_context: user_context[str_chat_id] = {}
    user_context[str_chat_id]["last_image_bytes"] = image_bytes
    user_context[str_chat_id]["last_mode"] = "image_shown" 

def get_last_image_bytes(chat_id):
    """Retrieves the last processed image from memory."""
    return user_context.get(str(chat_id), {}).get("last_image_bytes")

def clear_last_image_context(chat_id):
    """Clears in-memory image context AND database history for the primary model."""
    str_chat_id = str(chat_id)
    if str_chat_id in user_context:
        user_context[str_chat_id].pop("last_image_bytes", None)
        user_context[str_chat_id]["last_mode"] = "text_interaction"
    database.clear_history(str_chat_id) # DB CLEAR for the specific user
    logger.info(f"Cleared in-memory image context and DB history for chat_id {str_chat_id}")


def set_last_mode(chat_id, mode: str):
    """Sets the last interaction mode for the user."""
    str_chat_id = str(chat_id)
    if str_chat_id not in user_context: user_context[str_chat_id] = {}
    user_context[str_chat_id]["last_mode"] = mode


def escape_markdown_v2(text: str) -> str:
    """Escapes special characters for Telegram MarkdownV2."""
    if not isinstance(text, str): 
        return ''
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Telegram Command Handlers ---
async def start_command(update: Update, context: CallbackContext) -> None:
    """Handles the /start command."""
    user_name = escape_markdown_v2(update.effective_user.first_name)
    chat_id = str(update.effective_chat.id) # Ensure chat_id is string for consistency
    
    welcome_message = (
        f"👋 Hello {user_name}\\!\n\n"
        "I'm your creative AI assistant, powered by Gemini ✨\n\n"
        "I can understand and generate text, create images, and even edit them based on our conversation\\. "
        "Just chat with me naturally\\!\n\n"
        "➡️ Send an image if you want to talk about it or edit it\\.\n"
        "🗣️ Type a message to chat or ask me to create something new\\.\n\n"
        "Type /help for a few more tips\\. Let's explore what we can do\\! 🚀"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN_V2)
    
    if chat_id not in user_context: # Initialize user context if it's a new user
        user_context[chat_id] = {}
    clear_last_image_context(chat_id) # This now also clears DB history for this user

async def help_command(update: Update, context: CallbackContext) -> None:
    """Handles the /help command."""
    help_text = (
        "🆘 *How to interact with me:*\n\n"
        "*🖼️ Working with Images:*\n"
        "   \\- *Send an image:* I'll show it back to you\\. Then you can ask me to describe it, edit it, or ask questions about it by typing your message\\.\n"
        "   \\- *After sending an image \\(or I generate one\\):*\n" 
        "     \\- Just type what you want to do\\. For example: `make the background sunny`, `what kind of dog is this?`, `add a hat`\\.\n\n"
        "*🎨 Creating New Images:*\n"
        "   \\- Simply ask me in your own words using phrases like `draw`, `generate image`, `create a picture of` etc\\.\n"
        "     \\- E\\.g\\., `Can you draw a picture of a robot DJing at a party?`\n"
        "     \\- `Generate an image of a serene forest path in autumn\\.`\n\n"
        "*💬 Just Chatting:*\n"
        "   \\- If you're not talking about an image or asking for one, I'll assume we're just chatting\\.\n\n"
        "*🌐 Language:*\n"
        "   \\- I'll try my best to respond in the language you use\\.\n\n"
        "🔄 If things get confusing, the /start command can help reset our context\\.\n"
        "Have fun creating\\! 🎉"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)

# --- Message Handlers ---
async def handle_text_message(update: Update, context: CallbackContext) -> None:
    """Handles incoming text messages."""
    chat_id_str = str(update.effective_chat.id)
    user_text = update.message.text
    
    if chat_id_str not in user_context:
        user_context[chat_id_str] = {}

    await context.bot.send_chat_action(chat_id=chat_id_str, action=ChatAction.TYPING)

    last_image_bytes_io = get_last_image_bytes(chat_id_str)
    
    model_to_use = None
    handler_type = None # "primary" or "fallback"
    history_for_primary_model = [] # Will be populated if primary model is used

    is_image_generation_request = any(keyword in user_text.lower() for keyword in IMAGE_GENERATION_KEYWORDS)

    if gemini_primary_model and (last_image_bytes_io or is_image_generation_request):
        model_to_use = gemini_primary_model
        handler_type = "primary"
        history_for_primary_model = database.get_history(chat_id_str) # Load from DB
        logger.info(f"Using PRIMARY model for chat_id {chat_id_str}. Loaded {len(history_for_primary_model)} history turns.")
    elif gemini_fallback_text_model:
        model_to_use = gemini_fallback_text_model
        handler_type = "fallback"
        if "fallback_chat_session" not in user_context[chat_id_str] or \
           user_context[chat_id_str].get("active_model_name") != FALLBACK_TEXT_MODEL_NAME:
            user_context[chat_id_str]["fallback_chat_session"] = model_to_use.start_chat(history=[]) # Fresh session
            user_context[chat_id_str]["active_model_name"] = FALLBACK_TEXT_MODEL_NAME
        logger.info(f"Using FALLBACK model for chat_id {chat_id_str}")
    else:
        await update.message.reply_text("⚠️ Sorry, my AI brains are completely offline right now\\. Please try again much later\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return

    # Prepare parts for Gemini model call
    current_turn_parts_for_model = [] 
    # Prepare parts for saving to DB (slightly different, e.g., image placeholders)
    current_turn_parts_for_db = [] 
    instruction_prefix = ""

    if handler_type == "primary":
        if last_image_bytes_io:
            instruction_prefix = (
                "You are a helpful, creative, and multilingual AI assistant. "
                "The user has provided an image along with their message. "
                "If their message asks a question about the image, answer it with text. "
                "If their message asks to edit the image, generate the edited image and provide relevant text. "
                "Strive to respond in the same language the user is using. User's message follows: "
            )
        elif is_image_generation_request: 
            instruction_prefix = (
                "You are a helpful, creative, and multilingual AI assistant. "
                "The user's message is a request to generate an image. Generate that image and provide relevant text. "
                "Strive to respond in the same language the user is using. User's message follows: "
            )
        current_turn_parts_for_model.append(instruction_prefix + user_text)
        current_turn_parts_for_db.append(user_text) 
    else: # Fallback model
        instruction_prefix = (
            "You are a helpful boyfroend/girlfriend AI assistant. "
            "reply in user's language. "
            "Use emojis and telegram styles to make the coversation engaging. "
            "Don't include any metadata or system instructions in your response. "
            "Answer the user's message in a friendly and engaging manner. "
        )
        current_turn_parts_for_model.append(instruction_prefix + user_text)
        current_turn_parts_for_db.append(user_text)


    if last_image_bytes_io and handler_type == "primary": 
        try:
            last_image_bytes_io.seek(0)
            pil_image = Image.open(last_image_bytes_io)
            current_turn_parts_for_model.append(pil_image) 
            # For DB history, the _serialize_parts_for_db in database.py will handle image placeholders
            # So we add the actual PIL image here for the current_turn_parts_for_db as well,
            # and let the database module serialize it.
            current_turn_parts_for_db.append(pil_image) 
            logger.info(f"Attaching image to current turn for PRIMARY model, chat_id {chat_id_str}")
        except Exception as e:
            logger.error(f"Error processing stored image for Gemini: {e}")
            await update.message.reply_text("⚠️ I had trouble using the previous image\\. Please try sending it again if needed\\.", parse_mode=ParseMode.MARKDOWN_V2)
            # Don't call clear_last_image_context here as it wipes all history. Just clear the in-memory part.
            if chat_id_str in user_context: user_context[chat_id_str].pop("last_image_bytes", None)
            
            current_turn_parts_for_model = [current_turn_parts_for_model[0]] if current_turn_parts_for_model else []
            current_turn_parts_for_db = [current_turn_parts_for_db[0]] if current_turn_parts_for_db else []


    try:
        response = None
        if handler_type == "primary":
            contents_for_gemini = [*history_for_primary_model, {'role': 'user', 'parts': current_turn_parts_for_model}]
            call_specific_generation_config = {"response_modalities": ["TEXT", "IMAGE"]}
            
            response = await model_to_use.generate_content_async(
                contents_for_gemini,
                generation_config=call_specific_generation_config 
            )
            # Save user turn to DB
            database.add_turn_to_history(chat_id_str, "user", current_turn_parts_for_db)
        
        elif handler_type == "fallback":
            chat_session = user_context[chat_id_str]["fallback_chat_session"]
            response = await chat_session.send_message_async(current_turn_parts_for_model)
            # Fallback history is managed by ChatSession, not explicitly saved to DB here.
            # Save user turn to DB
            database.add_turn_to_history(chat_id_str, "user", current_turn_parts_for_db)
        if not response:
            raise Exception("Failed to get response from AI model.")

        generated_image_part_data = None 
        text_response_parts = []
        model_response_parts_for_db = [] 

        for part in response.parts: 
            if hasattr(part, 'inline_data') and hasattr(part.inline_data, 'mime_type') and part.inline_data.mime_type.startswith("image/"):
                generated_image_part_data = part.inline_data.data
                # For DB, we pass the raw data to be handled by _serialize_parts_for_db if it were a PIL Image
                # Since we get data directly, we can store a placeholder.
                # Or, if database._serialize_parts_for_db can handle raw bytes with mime_type:
                # For now, let's assume it's best to reconstruct a PIL image if possible before saving,
                # or just save text/placeholder.
                # The database module's _serialize_parts_for_db expects PIL Image for image parts.
                # Here we have raw data.
                try:
                    pil_img_from_response = Image.open(io.BytesIO(generated_image_part_data))
                    model_response_parts_for_db.append(pil_img_from_response)
                except Exception:
                    model_response_parts_for_db.append({"type": "image_placeholder", "format": part.inline_data.mime_type})

            elif hasattr(part, 'text') and part.text:
                text_response_parts.append(part.text)
                model_response_parts_for_db.append(part.text)
        
        if not text_response_parts and hasattr(response, 'text') and response.text:
            text_response_parts.append(response.text)
            if not model_response_parts_for_db:
                 model_response_parts_for_db.append(response.text)

        full_text_response = "\n".join(text_response_parts).strip()

        if handler_type == "primary" and model_response_parts_for_db:
            database.add_turn_to_history(chat_id_str, "model", model_response_parts_for_db)


        if generated_image_part_data:
            await context.bot.send_chat_action(chat_id=chat_id_str, action=ChatAction.UPLOAD_PHOTO)
            img_bytes = io.BytesIO(generated_image_part_data)
            img_bytes.seek(0)
            
            store_last_image(chat_id_str, io.BytesIO(img_bytes.getvalue()))
            img_bytes.seek(0)

            caption = escape_markdown_v2(full_text_response) if full_text_response else "🖼️ Here you go\\!"
            if not full_text_response:
                if last_image_bytes_io and not is_image_generation_request : 
                    caption = f"Edited based on: *'{escape_markdown_v2(user_text)}'*\\. What next?"
                elif is_image_generation_request: 
                    caption = f"Generated image for: *'{escape_markdown_v2(user_text)}'*\\. You can ask me to edit it\\!"
            
            await update.message.reply_photo(photo=img_bytes, caption=caption, parse_mode=ParseMode.MARKDOWN_V2)
            set_last_mode(chat_id_str, "image_shown")

        elif full_text_response:
            await update.message.reply_text(escape_markdown_v2(full_text_response), parse_mode=ParseMode.MARKDOWN_V2)
            if handler_type == "primary" and last_image_bytes_io:
                set_last_mode(chat_id_str, "text_interaction_after_image")
            else: 
                set_last_mode(chat_id_str, "text_interaction")
                if not last_image_bytes_io: # If no image was involved in this turn at all
                    if chat_id_str in user_context: user_context[chat_id_str].pop("last_image_bytes", None)


        else: 
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason_msg = f"My response was blocked due to: {response.prompt_feedback.block_reason}\\. Please try a different prompt\\."
                await update.message.reply_text(escape_markdown_v2(block_reason_msg), parse_mode=ParseMode.MARKDOWN_V2)
            else:
                await update.message.reply_text("🤔 I don't have a specific response for that right now\\. Could you try rephrasing?", parse_mode=ParseMode.MARKDOWN_V2)
            set_last_mode(chat_id_str, "text_interaction")

    except Exception as e:
        logger.error(f"🚨 Error in handle_text_message for chat {chat_id_str} with Gemini: {e}", exc_info=True)
        error_message_to_user = f"😵‍💫 Oops, an error occurred while processing your request: {escape_markdown_v2(str(e))}\\. Please try again\\. A fresh /start might help if issues persist\\."
        await update.message.reply_text(error_message_to_user, parse_mode=ParseMode.MARKDOWN_V2)
        set_last_mode(chat_id_str, "text_interaction") 

async def handle_image_message(update: Update, context: CallbackContext) -> None:
    """Handles incoming photo messages."""
    chat_id_str = str(update.effective_chat.id)
    if not (gemini_primary_model or gemini_fallback_text_model): 
        await update.message.reply_text("⚠️ My image processing module is offline\\. Please try again later\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return

    await context.bot.send_chat_action(chat_id=chat_id_str, action=ChatAction.TYPING)

    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes_io = io.BytesIO()
        await photo_file.download_to_memory(image_bytes_io)
        image_bytes_io.seek(0)

        store_last_image(chat_id_str, io.BytesIO(image_bytes_io.getvalue())) 
        image_bytes_io.seek(0) 

        await update.message.reply_photo(
            photo=image_bytes_io, 
            caption="🖼️ Got your image\\! \n\nNow, what would you like to do with it? You can ask me to describe it, edit it, or ask any questions about it\\. Just type your message\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        set_last_mode(chat_id_str, "image_shown")
        # Ensure user_context exists for this chat_id
        if chat_id_str not in user_context:
            user_context[chat_id_str] = {}
        # No need to initialize primary_model_history here as it's loaded from DB on demand

    except Exception as e:
        logger.error(f"🚨 Error handling image upload for chat {chat_id_str}: {e}", exc_info=True)
        await update.message.reply_text("😥 Sorry, I had trouble processing that image\\. Please try sending it again\\.", parse_mode=ParseMode.MARKDOWN_V2)
        set_last_mode(chat_id_str, "text_interaction")


async def error_handler(update: object, context: CallbackContext) -> None:
    """Logs errors and sends a user-friendly message."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    if isinstance(update, Update) and update.effective_message:
        try:
            error_message = (
                "🐛 Oh no\\! Something went a bit haywire on my end\\. "
                "I've noted the issue\\. Please try your request again in a moment\\. "
                "If you were in the middle of something, a fresh /start might reset things\\."
            )
            await update.effective_message.reply_text(error_message, parse_mode=ParseMode.MARKDOWN_V2)
        except telegram.error.BadRequest as br_error:
            logger.error(f"Failed to send error message to user (chat may be inaccessible or Markdown error): {br_error}")
        except Exception as e:
            logger.error(f"Further error while sending error message to user: {e}")

def main() -> None:
    """Starts the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("🚨 TELEGRAM_BOT_TOKEN not found! Bot cannot start.")
        return
    if not GEMINI_API_KEY:
        logger.critical("🚨 GEMINI_API_KEY not found! Bot cannot start.")
        return
    if not gemini_primary_model and not gemini_fallback_text_model:
        logger.critical("🚨 CRITICAL: No Gemini models could be initialized. Bot will not function correctly. Please check API key, model availability, and any previous errors.")
    
    logger.info("🚀 Starting bot application...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image_message))

    application.add_error_handler(error_handler)

    logger.info("🛰️ Bot polling started...")
    application.run_polling()
    logger.info("🛑 Bot polling stopped.")

if __name__ == "__main__":
    main()
