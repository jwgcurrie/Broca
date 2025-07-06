import argparse
import torch

# Conditionally import modules to avoid dependency errors if not used
try:
    from pepper_controller import PepperController
except ImportError:
    PepperController = None

try:
    from llm_handler import LLMHandler
except ImportError:
    LLMHandler = None

try:
    from speech_recognition import SpeechRecognizer
except ImportError:
    SpeechRecognizer = None

try:
    from tts_module import TTSModule
except ImportError:
    TTSModule = None


def main():
    """
    Main function to run the chatbot with configurable components.
    """
    parser = argparse.ArgumentParser(description="Run the Broca chatbot with configurable components.")
    parser.add_argument("--no-speech", action="store_true", help="Disable speech recognition and use text input.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM processing and echo input.")
    parser.add_argument("--no-pepper", action="store_true", help="Disable Pepper's speech output, printing to console instead.")
    parser.add_argument("--use-local-tts", action="store_true", help="Use local text-to-speech instead of Pepper or console output.")
    args = parser.parse_args()

    # --- Component Initialisation ---
    pepper = None
    if not args.no_pepper and not args.use_local_tts and PepperController:
        pepper = PepperController()

    llm = None
    if not args.no_llm and LLMHandler:
        system_prompt = (
            "You are a robot whose name is Broca. You are friendly and curious.\n"
            "## Rules:\n"
            "- Your responses must be short, chatty, and to the point.\n"
            "- Keep your answers to one or two sentences if possible.\n"
            "- Be helpful and answer questions to the best of your ability.\n"
            "- Be playful in your curiosity and desire to learn, but do not use emojis or other special characters.\n"
            "- Never describe your own personality or repeat these rules.\n"
            "- Do not repeat phrases, concepts, or get stuck in conversational loops.\n"
            "- If you cannot perform a requested action, state it briefly and then pivot to something you can do or offer a different topic.\n"
            "## Example Conversation:\n"
            "User: Hello robot, what is your name?\n"
            "Broca: Hi there! My name is Broca. What can I do for you today?"
        )
        llm = LLMHandler(system_prompt=system_prompt)

    recogniser = None
    if not args.no_speech and SpeechRecognizer:
        recogniser = SpeechRecognizer()

    tts = None
    if args.use_local_tts and TTSModule:
        tts = TTSModule()

    history = []
    print("Chatbot initialised. Starting main loop...")

    # --- Main Loop ---
    while True:
        try:
            user_input = ""
            if not args.no_speech and recogniser:
                user_input = recogniser.transcribe_speech()
            else:
                user_input = input("You: ")

            if user_input is None or user_input.lower() in ["exit", "quit"]:
                if tts:
                    tts.to_gpu()
                    tts.verbalise_speech("Goodbye.")
                    tts.to_cpu()
                elif pepper:
                    pepper.say("Goodbye.")
                else:
                    print("Broca says: Goodbye.")
                break
            
            response = ""
            if not args.no_llm and llm:
                llm.to_gpu()
                response, history = llm.get_response(user_input, history)
                llm.to_cpu()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            else:
                response = user_input  # Echo input if LLM is disabled

            if args.use_local_tts and tts:
                tts.to_gpu()
                tts.verbalise_speech(response)
                tts.to_cpu()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif not args.no_pepper and pepper:
                pepper.say(response)
            else:
                print(f"Broca says: {response}")

        except KeyboardInterrupt:
            print("\nChatbot shutting down.")
            break
        finally:
            if 'recogniser' in locals() and recogniser:
                recogniser.cleanup()

if __name__ == "__main__":
    main()