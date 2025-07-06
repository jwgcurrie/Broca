import argparse
import torch

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
    from tts_module import TTSModule as ParlerTTSModule
except ImportError:
    ParlerTTSModule = None
try:
    from speecht5_tts import SpeechT5TTSModule
except ImportError:
    SpeechT5TTSModule = None


def main():
    """
    Main function to run the chatbot with configurable components.
    """
    parser = argparse.ArgumentParser(description="Run the Broca chatbot with configurable components.")
    parser.add_argument("--no-speech", action="store_true", help="Disable speech recognition and use text input.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM processing and echo input.")
    parser.add_argument("--no-pepper", action="store_true", help="Disable Pepper's speech output, printing to console instead.")
    parser.add_argument("--use-local-tts", action="store_true", help="Enable local text-to-speech.")

    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=['parler', 'speecht5'],
        default='speecht5',
        help="Choose the local TTS engine to use (default: speecht5)."
    )
    args = parser.parse_args()

    pepper = None
    if not args.no_pepper and not args.use_local_tts and PepperController:
        pepper = PepperController()

    llm = None
    if not args.no_llm and LLMHandler:
        system_prompt = (
            "You are a robot named Broca. You are friendly and curious. Keep your answers to one or two sentences."
        )
        llm = LLMHandler(system_prompt=system_prompt)

    recogniser = None
    if not args.no_speech and SpeechRecognizer:
        recogniser = SpeechRecognizer()

    tts = None
    if args.use_local_tts:
        print(f"Selected TTS engine: {args.tts_engine}")
        if args.tts_engine == 'parler' and ParlerTTSModule:
            tts = ParlerTTSModule()
        elif args.tts_engine == 'speecht5' and SpeechT5TTSModule:
            tts = SpeechT5TTSModule()

        if tts is None:
            print(f"Warning: Could not initialise the '{args.tts_engine}' TTS engine.")

    history = []
    print("Chatbot initialised. Starting main loop...")

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
                response = user_input

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