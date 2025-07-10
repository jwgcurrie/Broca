# Broca: Built-in Robotic Offline Conversational Application

Broca is a foundational project for developing a local chatbot designed to interact onboard social robots like Pepper or iCub. This project focuses on integrating local Speech-to-Text, a Large Language Model (LLM), and Text-to-Speech to enable fully offline conversational capabilities without relying on cloud services.

### Table of Contents
* [Features](#features)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)

---

## Features

* **Local Speech Recognition**: Integrates an offline speech-to-text module using `arecord` and the Hugging Face Transformers `openai/whisper-base` model.
* **Local LLM Integration**: Utilises Hugging Face's `transformers` library to run a local LLM (e.g., `HuggingFaceTB/SmolLM2-360M-Instruct`).
* **Local Text-to-Speech (TTS)**: Provides verbal responses using local models. Includes a high-quality, balanced engine (**Microsoft SpeechT5**) and an expressive engine (**Parler-TTS**).
* **GPU Memory Management**: Implements a "workbench" strategy to shuttle models on and off the GPU, allowing multiple large models to run on a single, memory-constrained device.
* **Configurable Personality**: The chatbot's personality and rules can be easily customised via a system prompt.
* **Modular Design**: Separates concerns into `llm_handler.py`, `speech_recognition.py`, various `tts` modules, and `main.py` for orchestration.
* **Component Control**: Command-line arguments allow for granular control over speech recognition, LLM processing, and the choice of TTS engine.
* **Expressive Status Display**: Provides real-time visual feedback on the bot's state (idle, listening, thinking, speaking, error) directly in the terminal using animated, color-coded faces.

    | State      | Face Animation Frames       | Color  |
    | :--------- | :-------------------------- | :----- |
    | **Idle**   | `[ -_- ]` (Static)          | Grey   |
    | **Listening**| `[ o.o ]` (Static)          | Blue   |
    | **Thinking** | `[ o_O ]` <-> `[ O_o ]`     | Yellow |
    | **Speaking** | `[ ^o^ ]` <-> `[ ^-^ ]`     | Green  |
    | **Error**    | `[ x.! ]` <-> `[ !.x ]`     | Red    |

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

* **Python 3.11**: This version is recommended for best compatibility with the required machine learning libraries.
* **A Linux Environment**: Required for the `arecord` utility used in speech recognition. For Debian/Ubuntu, install with `sudo apt-get install alsa-utils`.
* **NVIDIA GPU with CUDA**: Highly recommended for acceptable performance.
* `pip` and `venv` (Python package and environment managers).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/jwgcurrie/Broca.git](https://github.com/jwgcurrie/Broca.git)
    cd Broca
    ```
2.  **Create a virtual environment**:
    ```bash
    python3.11 -m venv venv
    ```
3.  **Activate the virtual environment**:
    * On Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
4.  **Install dependencies**:
    The required packages are listed in the `requirements.txt` file included in the repository. Run:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the chatbot, activate your virtual environment and execute `main.py` from the `src` directory.

```bash
python3 src/main.py [OPTIONS]
```

### Options

* `--no-speech`: Disables voice recognition, requiring you to type your input manually.
* `--no-llm`: Disables the language model. The application will simply echo your input back as its "response".
* `--use-local-tts`: Enables the text-to-speech engine to speak responses aloud. Without this, responses are printed to the console.
* `--tts-engine <engine>`: Selects which TTS engine to use. Your choices are `speecht5` (the default) or `parler`.
* `--no-pepper`: Legacy flag to disable direct robot interaction. This is now the default behaviour.
* `--verbose`: Enable verbose output for debugging.

### Examples

* **Run with default settings (Speech Input, LLM, SpeechT5 TTS Output):**
    ```bash
    python3 src/main.py --use-local-tts
    ```
* **Run with the alternative Parler TTS engine:**
    ```bash
    python3 src/main.py --use-local-tts --tts-engine parler
    ```
* **Run with text input and SpeechT5 output:**
    ```bash
    python3 src/main.py --no-speech --use-local-tts
    ```

---

## Configuration

### LLM Model

The default LLM can be changed in `src/llm_handler.py` by modifying the `model_id` parameter. Be mindful of your VRAM limitations when choosing a larger model.

### Chatbot Personality

The chatbot's personality is defined by the `system_prompt` string in `src/main.py`. You can edit this to customise its behaviour and conversational style.

---

## Troubleshooting

* **`arecord: command not found`**: The `arecord` utility is missing. Install it on Debian/Ubuntu systems with `sudo apt-get install alsa-utils`.
* **Dependency Errors during `pip install`**: This project is tested on **Python 3.11**. If you encounter installation errors, ensure you are using a compatible Python version.
* **CUDA / GPU Errors**: Ensure your NVIDIA drivers and CUDA toolkit are installed correctly. If you have GPU issues, you can try running the models on the CPU by modifying the `device` variables in the respective module files, though performance will be significantly slower.