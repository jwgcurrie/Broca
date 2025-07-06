
import argparse

from pepper_controller import PepperController
from llm_handler import LLMHandler
from speech_recognition import SpeechRecognizer

def main():
    parser = argparse.ArgumentParser(description="Run the Broca chatbot with configurable components.")
    parser.add_argument("--no-speech", action="store_true", help="Disable speech recognition and use text input.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM processing and echo input.")
    parser.add_argument("--no-pepper", action="store_true", help="Disable Pepper's speech output.")
    args = parser.parse_args()
    """
    Main function to run the chatbot.
    """
    pepper = None
    if not args.no_pepper:
        pepper = PepperController()

    llm = None
    if not args.no_llm:
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

    speech_recognizer = None
    if not args.no_speech:
        speech_recognizer = SpeechRecognizer()
    
    history = []

    while True:
        try:
            user_input = ""
            if args.no_speech:
                user_input = input("You: ")
            else:
                user_input = speech_recognizer.transcribe_speech()
                if user_input is None: # Handle KeyboardInterrupt or errors in speech recognition
                    break
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            response = ""
            if args.no_llm:
                response = user_input # Echo input if LLM is disabled
            else:
                response, history = llm.get_response(user_input, history)
            
            if args.no_pepper:
                print(f"Broca says: {response}")
            else:
                pepper.say(response)

        except KeyboardInterrupt:
            print("\nChatbot shutting down.")
            if speech_recognizer:
                speech_recognizer.cleanup()
            break

if __name__ == "__main__":
    main()
