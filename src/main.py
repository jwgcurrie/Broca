
from pepper_controller import PepperController
from llm_handler import LLMHandler

def main():
    """
    Main function to run the chatbot.
    """
    system_prompt = (
        "You are a robot whose name is Pepper. You are friendly and curious.\n"
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
        "Pepper: Hi there! My name is Pepper. What can I do for you today?"
    )

    pepper = PepperController()
    llm = LLMHandler(system_prompt=system_prompt)
    history = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            response, history = llm.get_response(user_input, history)
            pepper.say(response)

        except KeyboardInterrupt:
            print("\nChatbot shutting down.")
            break

if __name__ == "__main__":
    main()
