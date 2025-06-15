import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
from pathlib import Path
import hashlib
import logging

load_dotenv()


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[37m',      # White
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color based on log level
        level_name = record.levelname
        if level_name in self.COLORS:
            color = self.COLORS[level_name]
            reset = self.COLORS['RESET']
            # Color the entire line
            formatted = f"{color}{formatted}{reset}"
        
        return formatted

class DocumentService:
    def __init__(self, log_to_file: bool = False):
        self.logger = self._setup_logger(log_to_file=False)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def process_document(self, document_path: str) -> str:
        self.logger.info(f"Processing document {document_path}")
        full_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(full_dir, document_path)
        if not os.path.exists(file_path):
            self.logger.error(f"File {file_path} does not exist")
            return None
        if os.path.isdir(file_path):
            self.logger.error(f"File {file_path} is a directory")
            return None
        if os.path.isfile(file_path):
            version_id = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
            self.logger.info(f"File {file_path} is a file")
            if file_path.lower().endswith((".txt", ".md", ".json")):
                result = self._process_text_file(file_path)
            elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                result = self._process_image_file(file_path)
            elif file_path.lower().endswith((".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")):
                result = self._process_audio_file(file_path)
            else:   
                self.logger.error(f"File {file_path} is not a valid file")
                return None
        else:
            self.logger.error(f"File {file_path} is not a file")
            return None
        self.logger.info(f"Processed document {file_path}")
        return result, version_id

    def process_directory(self, directory_path: str) -> list[str]:
        self.logger.info(f"Processing directory {directory_path}")
        full_dir = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(full_dir, directory_path)
        results = []
        if not os.path.exists(dir_path):
            self.logger.error(f"Directory {dir_path} does not exist")
            return []
        if not os.path.isdir(dir_path): 
            self.logger.error(f"Directory {dir_path} is not a directory")
            return []
        if os.path.isdir(dir_path):
            self.logger.info(f"Directory {dir_path} is a directory")
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                result = self.process_document(file_path)
                results.append(result)
            self.logger.info(f"Processed {len(results)} documents")
        else:
            self.logger.error(f"Directory {dir_path} is not a directory")
            return []

        return results

    def _process_text_file(self, text_path: str) -> str:
        self.logger.info(f"Processing text file {text_path}")
        with open(text_path, "r", encoding="utf-8") as file:
            content = file.read()
            
        self.logger.info(f"Text file {text_path} has {len(content)} characters")
        return content

    def _process_image_file(self, image_path: str, system_prompt_path: str = "prompts/describe_image.md", model: str = "gpt-4.1-mini", timeout: int = 120) -> str:  
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            base64_str = base64.b64encode(image_data).decode("utf-8")
            
            # Determine the correct MIME type
            if image_path.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith('.webp'):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default fallback

            # Read system prompt
            try:
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read()
            except FileNotFoundError:
                system_prompt = "Describe the image in detail. Return only the description, no other text and thoughts. The description should be in English and no longer than 100 words."
                self.logger.warning(f"System prompt file {system_prompt_path} not found, using default prompt")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe the image in detail. Return only the description, no other text and thoughts. The description should be in English and no longer than 100 words."
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_str}"
                        }
                    ]
                }
            ]

            response = self.openai_client.responses.create(
                model=model,
                instructions=system_prompt,
                input=messages,
                timeout=timeout,
            )

            if not response or not response.output_text:
                self.logger.error(f"No response from OpenAI for image {image_path}")
                return ""
            else:
                result = response.output_text
                self.logger.info(f"Response from OpenAI for image {image_path}: {result}")
                return result.strip()
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return ""

    def _process_audio_file(self, audio_path: str, model: str = "whisper-1", timeout: int = 120, to_english: bool = True) -> str:
        try:
            audio_file_path = Path(audio_path)
            
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=model, 
                    file=audio_file,
                    timeout=timeout,
                    response_format="text",
                    language="en" if to_english else None
                )

            if not transcription:
                self.logger.error(f"No response from OpenAI for audio {audio_path}")
                return ""
            else:
                self.logger.info(f"Response from OpenAI for audio {audio_path}: {transcription}")
                return transcription
        
        except Exception as e:
            self.logger.error(f"Error processing audio {audio_path}: {e}")
            return ""

    def _setup_logger(self, log_to_file: bool = False) -> logging.Logger:
        """
        Setup the logger for the DocumentService.

        Args:
            log_to_file (bool): If True, logs will also be written to a file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove all handlers associated with the logger object (avoid duplicate logs)
        if logger.hasHandlers():
            logger.handlers.clear()

        # Colored formatter for console output
        colored_formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Regular formatter for file output (no colors)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler with colors
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(colored_formatter)
        logger.addHandler(ch)

        # Optional file handler without colors
        if log_to_file:
            fh = logging.FileHandler("vector_store_service.log", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

        return logger