"""Run the ai-audio-toolkit application directly."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from audio_editor.main import main

if __name__ == "__main__":
    main()
