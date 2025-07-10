import time
import threading

class FaceDisplay:
    """
    Manages an expressive, animated status display for the bot in the terminal.
    """
    def __init__(self):
        self._state = 'idle'
        self._paused = False
        self._state_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Design Specification with animation frames
        self.faces = {
            'idle': ['[ -_- ]'],
            'listening': ['[ o.o ]'],
            'thinking': ['[ o_O ]', '[ O_o ]'],
            'speaking': ['[ ^o^ ]', '[ ^-^ ]'],
            'error': ['[ x.! ]', '[ !.x ]']
        }
        self.term_colors = {
            'idle': '\033[90m',      # Grey
            'listening': '\033[94m',  # Blue
            'thinking': '\033[93m',  # Yellow
            'speaking': '\033[92m',  # Green
            'error': '\033[91m',      # Red
            'end': '\033[0m',
        }

        # Start the animation thread
        self.animation_thread = threading.Thread(target=self._animation_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def set_state(self, new_state):
        """Thread-safely sets the bot's current state."""
        with self._state_lock:
            self._state = new_state

    def get_state(self):
        """Thread-safely gets the bot's current state."""
        with self._state_lock:
            return self._state

    def pause(self):
        """Pauses the animation thread."""
        with self._state_lock:
            self._paused = True

    def resume(self):
        """Resumes the animation thread."""
        with self._state_lock:
            self._paused = False

    def _animation_loop(self):
        """The main rendering loop, run in a separate thread."""
        frame_index = 0
        while not self.shutdown_event.is_set():
            with self._state_lock:
                if self._paused:
                    time.sleep(0.1) # Sleep briefly while paused
                    continue

            current_state = self.get_state()
            # Default to idle state if the state is unknown
            face_frames = self.faces.get(current_state, self.faces['idle'])

            # Cycle through animation frames
            if frame_index >= len(face_frames):
                frame_index = 0
            current_face = face_frames[frame_index]

            self._update_terminal(current_state, current_face)

            frame_index += 1
            time.sleep(0.5)  # Animation speed

    def _update_terminal(self, state, face):
        """Updates the status line in the terminal."""
        # Default to idle color if the state is unknown
        color = self.term_colors.get(state, self.term_colors['idle'])
        # For unknown states, use the state name itself
        state_name = state if state in self.faces else 'idle'
        status_text = f"{face} {state_name.capitalize()}..."
        
        # Print with carriage return and flush to ensure it updates immediately
        print(f"\r{color}{status_text.ljust(40)}{self.term_colors['end']}", end="", flush=True)

    def cleanup(self):
        """Shuts down the animation thread and cleans up resources."""
        self.shutdown_event.set()
        self.animation_thread.join(timeout=1)
        print()  # Print a final newline

if __name__ == '__main__':
    # Test harness to demonstrate the animated FaceDisplay class
    face = FaceDisplay()
    print("--- Testing Animated FaceDisplay Terminal Output ---")
    print("Cycling through states...")
    try:
        states = ['listening', 'thinking', 'speaking', 'idle', 'error']
        for state in states:
            print(f"\nSetting state to: {state}")
            face.set_state(state)
            # Let the animation run for a few seconds
            time.sleep(4) 

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        face.cleanup()
        print("--- Test Complete ---")