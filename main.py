import logging
import os
import io
from PIL import Image
import re
import html # For HTML escaping
import tempfile # For temporary HTML file

import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram.constants import ChatAction, ParseMode
import google.api_core.exceptions # Import for specific exception handling

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig, GenerateContentResponse

from dotenv import load_dotenv

# --- Database Utilities ---
import database # Your database module

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

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME", "gemini-2.0-flash-preview-image-generation")
FALLBACK_TEXT_MODEL_NAME = os.getenv("FALLBACK_TEXT_MODEL_NAME", "gemini-2.0-flash")

FALLBACK_SYSTEM_INSTRUCTION = (
    "You are a helpful boyfriend/girlfriend AI assistant. "
    "Reply in the user's language. Use emojis and Telegram styles. "
    "Do not include metadata or your persona description. "
    "Answer in a friendly, engaging manner."
)

gemini_primary_model = None
gemini_fallback_text_model = None

try:
    gemini_primary_model = genai.GenerativeModel(PRIMARY_MODEL_NAME, safety_settings=SAFETY_SETTINGS)
    logger.info(f"✅ Initialized primary model: {PRIMARY_MODEL_NAME}")
except Exception as e:
    logger.error(f"🚨 Error initializing primary model ({PRIMARY_MODEL_NAME}): {e}")

try:
    gemini_fallback_text_model = genai.GenerativeModel(
        FALLBACK_TEXT_MODEL_NAME,
        safety_settings=SAFETY_SETTINGS,
        system_instruction=FALLBACK_SYSTEM_INSTRUCTION
    )
    logger.info(f"✅ Initialized fallback model: {FALLBACK_TEXT_MODEL_NAME}")
except Exception as e:
    logger.error(f"🚨 Error initializing fallback model ({FALLBACK_TEXT_MODEL_NAME}): {e}")

user_context = {}
IMAGE_GENERATION_KEYWORDS = [
    "draw", "generate image", "create a picture", "make an image", "show me an image of",
    "picture of", "image of", "photo of", "illustration of", "sketch of", "design an image"
]

def store_last_image(chat_id, image_bytes: io.BytesIO):
    str_chat_id = str(chat_id)
    if str_chat_id not in user_context: user_context[str_chat_id] = {}
    user_context[str_chat_id]["last_image_bytes"] = image_bytes
    user_context[str_chat_id]["last_mode"] = "image_shown"

def get_last_image_bytes(chat_id) -> io.BytesIO | None:
    return user_context.get(str(chat_id), {}).get("last_image_bytes")

def clear_last_image_context(chat_id):
    str_chat_id = str(chat_id)
    if str_chat_id in user_context:
        user_context[str_chat_id].pop("last_image_bytes", None)
        user_context[str_chat_id]["last_mode"] = "text_interaction"
        user_context[str_chat_id].pop("fallback_chat_session", None)
        user_context[str_chat_id].pop("active_model_name", None)
    database.clear_history(str_chat_id)
    logger.info(f"Cleared context and DB history for chat_id {str_chat_id}")

def set_last_mode(chat_id, mode: str):
    str_chat_id = str(chat_id)
    if str_chat_id not in user_context: user_context[str_chat_id] = {}
    user_context[str_chat_id]["last_mode"] = mode

def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        return ''
    
    temp_text = text
    # Step 1: Heuristically "un-escape" common LaTeX-like sequences the model might output.
    # This converts model's attempts at escaping (e.g., "\(") into raw characters (e.g., "("),
    # so our comprehensive escaper in Step 2 can apply the correct Telegram MarkdownV2 escaping.
    model_specific_unescapes = {
        r'\(': '(', r'\)': ')',
        r'\[': '[', r'\]': ']',
        r'\{': '{', r'\}': '}',
        r'\_': '_', r'\*': '*',
        r'\~': '~', r'\`': '`',
        r'\.': '.', r'\!': '!',
        r'\#': '#', r'\+': '+',
        r'\-': '-', r'\=': '=',
        r'\|': '|', r'\>': '>',
    }
    for model_esc, raw_char in model_specific_unescapes.items():
        temp_text = temp_text.replace(model_esc, raw_char)
    
    # Step 2: Apply standard MarkdownV2 escaping to the "normalized" text.
    # Characters that MUST be escaped in MarkdownV2 to be treated as literal.
    final_escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(final_escape_chars)}])', r'\\\1', temp_text)


async def start_command(update: Update, context: CallbackContext) -> None:
    user_name = escape_markdown_v2(update.effective_user.first_name)
    chat_id_str = str(update.effective_chat.id)
    welcome_message = (
        f"👋 Hello {user_name}\\!\n\n"
        "I'm your creative AI assistant, powered by Gemini ✨\n\n"
        "I can understand and generate text, create images, and even edit them based on our conversation\\. "
        "Just chat with me naturally\\!\n\n"
        "➡️ Send an image if you want to talk about it or edit it\\.\n"
        "🗣️ Type a message to chat or ask me to create something new\\.\n\n"
        "Type /help for a few more tips\\. Let's explore what we can do\\! 🚀\n\n"
        "Using /start clears our previous conversation history to begin fresh\\."
    )
    # For static messages like this, ensure any manual MarkdownV2 is correct.
    # This one is already correctly pre-escaped.
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN_V2)
    if chat_id_str not in user_context:
        user_context[chat_id_str] = {}
    clear_last_image_context(chat_id_str)
    logger.info(f"Context cleared and initialized for chat_id {chat_id_str} on /start")


async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "🆘 *How to interact with me:*\n\n"
        "*🖼️ Working with Images:*\n"
        "  \\- *Send an image:* I'll show it back to you\\. Then you can ask me to describe it, edit it, or ask questions about it by typing your message\\.\n"
        "  \\- *After sending an image \\(or I generate one\\):*\n"
        "    \\- Just type what you want to do\\. For example: `make the background sunny`, `what kind of dog is this?`, `add a hat`\\.\n\n"
        "*🎨 Creating New Images:*\n"
        "  \\- Simply ask me in your own words using phrases like `draw`, `generate image`, `create a picture of` etc\\.\n"
        "    \\- E\\.g\\., `Can you draw a picture of a robot DJing at a party?`\n"
        "    \\- `Generate an image of a serene forest path in autumn\\.`\n\n"
        "*💬 Just Chatting:*\n"
        "  \\- If you're not talking about an image or asking for one, I'll assume we're just chatting \\(using my friendly persona\\)\\.\n\n"
        "*🌐 Language:*\n"
        "  \\- I'll try my best to respond in the language you use\\.\n\n"
        "🔄 If things get confusing, the /start command can help reset our context\\.\n"
        "Have fun creating\\! 🎉"
    )
    # This static message is also pre-escaped correctly.
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_text_message(update: Update, context: CallbackContext) -> None:
    chat_id_str = str(update.effective_chat.id)
    user_text = update.message.text
    if not user_text: return
    logger.info(f"Received text from {chat_id_str}: '{user_text[:50]}...'")
    if chat_id_str not in user_context: user_context[chat_id_str] = {}
    await context.bot.send_chat_action(chat_id=chat_id_str, action=ChatAction.TYPING)

    last_image_bytes_io = get_last_image_bytes(chat_id_str)
    model_to_use, handler_type, history_for_primary_model_raw = None, None, []
    is_image_generation_request = any(keyword in user_text.lower() for keyword in IMAGE_GENERATION_KEYWORDS)

    if gemini_primary_model and (last_image_bytes_io or is_image_generation_request):
        model_to_use, handler_type = gemini_primary_model, "primary"
        history_for_primary_model_raw = database.get_history(chat_id_str)
        logger.info(f"PRIMARY model. Loaded {len(history_for_primary_model_raw)} raw history turns for chat {chat_id_str}.")
        user_context[chat_id_str]["active_model_name"] = PRIMARY_MODEL_NAME
    elif gemini_fallback_text_model:
        model_to_use, handler_type = gemini_fallback_text_model, "fallback"
        if "fallback_chat_session" not in user_context[chat_id_str] or \
           user_context[chat_id_str].get("active_model_name") != FALLBACK_TEXT_MODEL_NAME:
            logger.info(f"Starting new fallback chat session for chat_id {chat_id_str}.")
            user_context[chat_id_str]["fallback_chat_session"] = model_to_use.start_chat(history=[])
            user_context[chat_id_str]["active_model_name"] = FALLBACK_TEXT_MODEL_NAME
        else:
            logger.info(f"Continuing fallback chat session for chat_id {chat_id_str}.")
    else:
        await update.message.reply_text(escape_markdown_v2("⚠️ AI brains are offline. Please try again later."), parse_mode=ParseMode.MARKDOWN_V2)
        return

    current_turn_parts_for_model, current_turn_parts_for_db = [], [user_text]
    if handler_type == "primary":
        instruction_prefix = ""
        if last_image_bytes_io: # Simplified instruction prefix
            instruction_prefix = "User provided an image. Their message is about this image. If it's a question, answer with text. If it's an edit request, generate the edited image and provide relevant text. Respond in user's language. User's message: "
            current_turn_parts_for_model.append(instruction_prefix + user_text)
            try:
                last_image_bytes_io.seek(0)
                current_turn_parts_for_model.append(Image.open(last_image_bytes_io))
            except Exception as e:
                logger.error(f"Error processing stored image for primary model: {e}")
                await update.message.reply_text(escape_markdown_v2("⚠️ Had trouble using the previous image. Please try sending it again if needed."), parse_mode=ParseMode.MARKDOWN_V2)
                if chat_id_str in user_context: user_context[chat_id_str].pop("last_image_bytes", None)
                current_turn_parts_for_model = [instruction_prefix + user_text] # Fallback to text only for this turn
        elif is_image_generation_request:
            instruction_prefix = "User's message is a request to generate an image. Generate that image and provide relevant text. Respond in user's language. User's message: "
            current_turn_parts_for_model.append(instruction_prefix + user_text)
        else: # Should ideally not happen for primary if no image context or gen request
            current_turn_parts_for_model.append(user_text)
    else: # Fallback
        current_turn_parts_for_model.append(user_text)

    database.add_turn_to_history(chat_id_str, "user", current_turn_parts_for_db)

    try:
        response: GenerateContentResponse | None = None
        if handler_type == "primary":
            processed_history_for_gemini = []
            for turn in history_for_primary_model_raw:
                new_turn_parts = []
                db_parts = turn.get('parts', [])
                if not isinstance(db_parts, list): db_parts = [str(db_parts)]
                for part_content in db_parts:
                    if isinstance(part_content, dict) and part_content.get('type') == 'image_placeholder':
                        img_desc = f"[Image: {part_content.get('format', 'ukn')}, size {part_content.get('size', 'ukn')}. This was part of a previous turn.]"
                        new_turn_parts.append(img_desc)
                    elif isinstance(part_content, str):
                        new_turn_parts.append(part_content)
                    else:
                        new_turn_parts.append(str(part_content))
                if new_turn_parts: processed_history_for_gemini.append({'role': turn['role'], 'parts': new_turn_parts})
            
            contents_for_gemini = [*processed_history_for_gemini, {'role': 'user', 'parts': current_turn_parts_for_model}]
            logger.debug(f"Primary model contents for chat {chat_id_str} (processed history an current turn): {str(contents_for_gemini)[:500]}...")
            response = await model_to_use.generate_content_async(
                contents_for_gemini,
                generation_config={"response_modalities": ["TEXT", "IMAGE"]} # Specific to your preview model
            )
        elif handler_type == "fallback":
            chat_session = user_context[chat_id_str]["fallback_chat_session"]
            logger.debug(f"Sending to fallback model for chat {chat_id_str}: '{current_turn_parts_for_model}'")
            response = await chat_session.send_message_async(current_turn_parts_for_model)

        if not response: raise Exception("No response received from the AI model.")

        generated_image_part_data, text_response_parts, model_response_parts_for_db_primary = None, [], []
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith("image/"):
                generated_image_part_data = part.inline_data.data
                if handler_type == "primary":
                    try: model_response_parts_for_db_primary.append(Image.open(io.BytesIO(generated_image_part_data)))
                    except Exception as e: logger.error(f"Error converting generated image part to PIL: {e}")
            elif hasattr(part, 'text') and part.text:
                text_response_parts.append(part.text)
                if handler_type == "primary": model_response_parts_for_db_primary.append(part.text)
        if not text_response_parts and hasattr(response, 'text') and response.text: # Fallback for simple text response
            text_response_parts.append(response.text)
            if handler_type == "primary" and not model_response_parts_for_db_primary: model_response_parts_for_db_primary.append(response.text)
        
        full_text_response = "\n".join(text_response_parts).strip()
        logger.info(f"--- RAW full_text_response from model (chat_id {chat_id_str}) --- \n'{full_text_response}'\n--------------------")
        if handler_type == "primary" and model_response_parts_for_db_primary:
            database.add_turn_to_history(chat_id_str, "model", model_response_parts_for_db_primary)

        if generated_image_part_data:
            await context.bot.send_chat_action(chat_id=chat_id_str, action=ChatAction.UPLOAD_PHOTO)
            img_bytes_to_send = io.BytesIO(generated_image_part_data)
            store_last_image(chat_id_str, io.BytesIO(img_bytes_to_send.getvalue())) # Store a copy
            img_bytes_to_send.seek(0)
            
            caption_raw = full_text_response # Use model's text as caption if available
            if not caption_raw: # Fallback caption
                 if last_image_bytes_io and not is_image_generation_request : caption_raw = f"Edited based on: '{user_text}'. What next?"
                 elif is_image_generation_request: caption_raw = f"Generated image for: '{user_text}'. You can ask me to edit it!"
                 else: caption_raw = "🖼️ Here's the image!"
            
            logger.info(f"--- RAW caption for photo (chat_id {chat_id_str}) --- \n'{caption_raw}'\n--------------------")
            try:
                escaped_caption = escape_markdown_v2(caption_raw)
                logger.info(f"--- ESCAPED caption for photo (chat_id {chat_id_str}) --- \n'{escaped_caption}'\n--------------------")
                await update.message.reply_photo(photo=img_bytes_to_send, caption=escaped_caption, parse_mode=ParseMode.MARKDOWN_V2)
            except telegram.error.BadRequest as br_caption_exc:
                logger.error(f"🚨 BadRequest sending photo caption with MarkdownV2 (chat_id {chat_id_str}): {br_caption_exc}. Raw caption: '{caption_raw}'", exc_info=True)
                await update.message.reply_photo(photo=img_bytes_to_send) # Send photo without caption
                if caption_raw: # Then send caption as plain text
                    await update.message.reply_text(f"(Caption for the image above as Markdown failed)\n{caption_raw}")
            set_last_mode(chat_id_str, "image_shown")

        elif full_text_response:
            try:
                escaped_response = escape_markdown_v2(full_text_response)
                logger.info(f"--- ESCAPED full_text_response for Telegram (chat_id {chat_id_str}) --- \n'{escaped_response}'\n--------------------")
                await update.message.reply_text(escaped_response, parse_mode=ParseMode.MARKDOWN_V2)
            except telegram.error.BadRequest as br_text_exc:
                logger.error(f"🚨 BadRequest sending reply_text with MarkdownV2 (chat_id {chat_id_str}): {br_text_exc}. Raw response: '{full_text_response}'", exc_info=True)
                try: # HTML Fallback
                    escaped_html_response = html.escape(full_text_response)
                    html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Bot Response</title><style>body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;line-height:1.6;margin:0;padding:20px;background-color:#eef2f7;color:#333;}} .container{{max-width:800px;margin:20px auto;background-color:#fff;padding:25px 30px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);}} h3{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;margin-top:0;}} .info-text{{font-style:italic;color:#7f8c8d;margin-bottom:15px;font-size:0.9em;}} pre{{background-color:#f8f9fa;padding:20px;border:1px solid #dee2e6;border-radius:6px;white-space:pre-wrap;word-wrap:break-word;font-family:'Courier New',Courier,monospace;font-size:0.95em;color:#212529;}} .footer{{text-align:center;margin-top:30px;font-size:0.8em;color:#95a5a6;}}</style></head><body><div class="container"><h3>Model's Response</h3><p class="info-text">This content could not be displayed directly in Telegram using Markdown formatting due to an issue. Please find the full response below.</p><pre><code>{escaped_html_response}</code></pre></div><p class="footer">Response processed by AI Assistant</p></body></html>"""
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding='utf-8') as tmp_f:
                        tmp_f.write(html_content)
                        tmp_f_path = tmp_f.name
                    logger.info(f"Created temporary HTML file for fallback: {tmp_f_path}")
                    await context.bot.send_document(chat_id=chat_id_str, document=open(tmp_f_path, 'rb'),
                                                    filename="response.html",
                                                    caption="There was an issue formatting the AI's response. Here it is as an HTML file.")
                    if os.path.exists(tmp_f_path):
                        os.remove(tmp_f_path)
                        logger.info(f"Deleted temporary HTML file: {tmp_f_path}")
                except Exception as html_fallback_exc:
                    logger.error(f"🚨🚨 Error during HTML fallback process (chat_id {chat_id_str}): {html_fallback_exc}", exc_info=True)
                    await update.message.reply_text(f"(Formatting Error: {br_text_exc})\nRaw content: {full_text_response}") # Ultimate fallback

            if handler_type == "primary" and last_image_bytes_io: set_last_mode(chat_id_str, "text_interaction_after_image")
            else: set_last_mode(chat_id_str, "text_interaction")
            if not last_image_bytes_io and handler_type == "primary": user_context[chat_id_str].pop("last_image_bytes", None)
        else: # No image and no text from model
            no_response_message_raw = ""
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                no_response_message_raw = f"My response was blocked due to: {response.prompt_feedback.block_reason}. Please try a different prompt or check safety settings."
                logger.warning(f"Response blocked for chat_id {chat_id_str}: {response.prompt_feedback.block_reason}")
            else:
                no_response_message_raw = "🤔 I don't have a specific response for that right now. Could you try rephrasing or a different request?"
                logger.warning(f"Empty response from model for chat_id {chat_id_str}. Response object: {response}")
            await update.message.reply_text(escape_markdown_v2(no_response_message_raw), parse_mode=ParseMode.MARKDOWN_V2)
            set_last_mode(chat_id_str, "text_interaction")

    except google.api_core.exceptions.InvalidArgument as e:
        logger.error(f"🚨 InvalidArgument Error (API call issue) for chat {chat_id_str}: {e}", exc_info=True)
        raw_msg = f"😵‍💫 AI request issue: *{str(e)}*.\nThis might be due to input data or model call. Try /start."
        await update.message.reply_text(escape_markdown_v2(raw_msg), parse_mode=ParseMode.MARKDOWN_V2)
        set_last_mode(chat_id_str, "text_interaction")
    except Exception as e:
        logger.error(f"🚨 General Error in handle_text_message for chat {chat_id_str}: {e}", exc_info=True)
        raw_msg = f"😵‍💫 Oops, an error occurred: *{str(e)}*.\nPlease try again. If the problem persists, a /start command might help."
        await update.message.reply_text(escape_markdown_v2(raw_msg), parse_mode=ParseMode.MARKDOWN_V2)
        set_last_mode(chat_id_str, "text_interaction")

async def handle_image_message(update: Update, context: CallbackContext) -> None:
    chat_id_str = str(update.effective_chat.id)
    logger.info(f"Received image from chat_id {chat_id_str}")
    if not (gemini_primary_model or gemini_fallback_text_model):
        await update.message.reply_text(escape_markdown_v2("⚠️ My AI core is offline. Please try again later."), parse_mode=ParseMode.MARKDOWN_V2)
        return
    await context.bot.send_chat_action(chat_id=chat_id_str, action=ChatAction.TYPING)
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes_io = io.BytesIO()
        await photo_file.download_to_memory(image_bytes_io)
        image_bytes_io.seek(0)
        store_last_image(chat_id_str, io.BytesIO(image_bytes_io.getvalue()))
        image_bytes_io.seek(0)
        database.add_turn_to_history(chat_id_str, "user", ["[User sent an image]"])
        caption_raw = "🖼️ Got your image! \n\nNow, what would you like to do with it? You can ask me to describe it, edit it, or ask any questions about it. Just type your message."
        await update.message.reply_photo(
            photo=image_bytes_io,
            caption=escape_markdown_v2(caption_raw),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        set_last_mode(chat_id_str, "image_shown")
        if chat_id_str not in user_context: user_context[chat_id_str] = {}
    except Exception as e:
        logger.error(f"🚨 Error handling image upload for chat {chat_id_str}: {e}", exc_info=True)
        await update.message.reply_text(escape_markdown_v2("😥 Sorry, I had trouble processing that image. Please try sending it again."), parse_mode=ParseMode.MARKDOWN_V2)
        set_last_mode(chat_id_str, "text_interaction")

async def error_handler(update: object, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if isinstance(context.error, telegram.error.NetworkError):
        logger.warning("NetworkError encountered. Bot might be having trouble connecting to Telegram.")
        return
    if isinstance(update, Update) and update.effective_message:
        try:
            error_message_raw = ( # This message has no special MarkdownV2 characters
                "🐛 Oh no! Something went a bit haywire on my end. "
                "I've noted the issue. Please try your request again in a moment. "
                "If you were in the middle of something, a fresh /start might reset things."
            )
            await update.effective_message.reply_text(error_message_raw, parse_mode=None) # Send as plain text
        except Exception as e:
            logger.error(f"Further error while sending error_handler message to user: {e}")

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("🚨 TELEGRAM_BOT_TOKEN not found! Bot cannot start.")
        return
    if not GEMINI_API_KEY:
        logger.critical("🚨 GEMINI_API_KEY not found! Bot cannot start.")
        return
    try:
        database.ensure_db_directory_exists()
    except Exception as e:
        logger.critical(f"🚨 CRITICAL: Could not create/access DB directory: {e}.")
    if not gemini_primary_model and not gemini_fallback_text_model:
        logger.critical("🚨 CRITICAL: No Gemini models initialized. Bot will not function.")
        return

    logger.info("🚀 Starting bot application...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image_message))
    application.add_error_handler(error_handler)
    logger.info("🛰️ Bot polling started...")
    try:
        application.run_polling()
    except Exception as e:
        logger.critical(f"🚨 Bot polling failed critically: {e}", exc_info=True)
    finally:
        logger.info("🛑 Bot polling stopped.")

if __name__ == "__main__":
    main()