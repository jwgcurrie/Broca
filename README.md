# Broca: **B**uilt-in **R**obotic **O**ffline **C**onversational **A**pplication


Broca is a foundational project for developing a local chatbot designed to interact with the Pepper robot. This project focuses on integrating a small local Large Language Model (LLM) to enable conversational capabilities without relying on external cloud services.

## Features

*   **Speech Recognition**: Integrates a local speech-to-text module using `arecord` and Hugging Face Transformers (Whisper model) for voice input.
*   **Local LLM Integration**: Utilizes Hugging Face's `transformers` library to run a local LLM (currently `HuggingFaceTB/SmolLM2-360M-Instruct`).
*   **Configurable Personality**: The chatbot's personality can be customized via a system prompt.
*   **Conversation History**: Maintains a short-term memory of the conversation for more coherent interactions.
*   **Modular Design**: Separates concerns into `llm_handler.py` (LLM interaction), `pepper_controller.py` (robot interaction), `speech_recognition.py` (speech-to-text), and `main.py` (orchestration).
*   **Component Control**: Command-line arguments allow for individual control over speech recognition, LLM processing, and Pepper's speech output.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/jwgcurrie/broca.git
    cd broca
    ```
2.  **Create a virtual environment**:
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv broca-py
    ```
3.  **Activate the virtual environment**:
    *   On Linux/macOS:
        ```bash
        source ./broca-py/bin/activate
        ```
    *   On Windows:
        ```bash
        .\broca-py\Scripts\activate
        ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the chatbot, activate your virtual environment and execute the `main.py` script:

```bash
source broca-py/bin/activate
python3 src/main.py [OPTIONS]
```

**Options:**

*   `--no-speech`: Disable speech recognition and use manual text input.
*   `--no-llm`: Disable LLM processing and echo input.
*   `--no-pepper`: Disable Pepper's speech output (responses will be printed to the console).

**Examples:**

*   Run with default settings (speech input, LLM processing, Pepper output):
    `python3 src/main.py`
*   Run with text input (no speech recognition):
    `python3 src/main.py --no-speech`
*   Run with speech input but no LLM processing (will echo what you say):
    `python3 src/main.py --no-llm`
*   Run with speech input and LLM, but no Pepper speech output (will print Broca's response):
    `python3 src/main.py --no-pepper`
*   Run with all components disabled (text input, echo, print response):
    `python3 src/main.py --no-speech --no-llm --no-pepper`


The chatbot will start, and you can begin interacting with it. Type `exit` or `quit` to end the conversation.

## Configuration

### LLM Model

The default LLM model is `HuggingFaceTB/SmolLM2-360M-Instruct`. You can change this in `src/llm_handler.py` by modifying the `model_id` parameter in the `LLMHandler`'s `__init__` method. Be mindful of your hardware's VRAM limitations when choosing a larger model.

### Chatbot Personality

The chatbot's personality is defined by the `system_prompt` variable in `src/main.py`. You can edit this string to customize Pepper's behavior and conversational style.

## Future Enhancements

*   **Full Pepper Robot Integration**: Implement actual communication with the Pepper robot via the NAOqi SDK for text-to-speech and other functionalities.
*   **Speech-to-Text (STT)**: Add a component for converting spoken input into text.
*   **Advanced Conversation Management**: Explore more sophisticated memory mechanisms for longer and more complex dialogues.
*   **Error Handling and Robustness**: Improve error handling and make the application more resilient to unexpected inputs or model behaviors.
