"""
Thin CLI demo for the Reply-Intent Classifier (Grok-powered).

Usage:
    python cli.py "What's your rate for a story post?"

If no argument is given, drops into an interactive loop.
"""

import sys
from classifier import classify_reply


def run(message: str):
    result = classify_reply(message)
    print(f"\nMessage:    {message}")
    print(f"Intent:     {result['intent']}")
    print(f"Confidence: {result['confidence']:.2f}\n")


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        run(message)
        return

    print("Reply-Intent Classifier — interactive mode (Ctrl+C to quit)")
    while True:
        try:
            message = input("\nPaste a creator reply: ").strip()
            if not message:
                continue
            run(message)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()