# =============================================================================
# main.py — Shapatar Bot Terminal Interface
# =============================================================================
#
# Terminal-based conversation loop. Thin shell around brain.py.
# Responsibilities:
#   - Start conversation (initialise context)
#   - Run the input/output event loop
#   - Pass input to brain, display output
#   - Handle errors and exit cleanly
#
# This file knows nothing about personality, mood logic, or responses.
#
# To run:   python3 main.py
# To exit:  type 'quit' or 'exit', or press Ctrl+C
#
# NOTE: The import below triggers model loading via brain.py's module-level
# initialise call. Loading output appears before the welcome banner.
# This is intentional — model is ready before conversation begins.
# =============================================================================

from brain import create_context, process_message


BOT_NAME   = "Shapatar Bot"
SEPARATOR  = "─" * 50
USER_LABEL = "You"
BOT_LABEL  = "Bot"


def print_header():
    """Print opening banner after model has loaded."""
    print("=" * 50)
    print(f"   {BOT_NAME}")
    print(f"   Semantic intent detection active.")
    print(f"   Type 'quit' or 'exit' to end.")
    print("=" * 50)
    print()


def print_turn(user_text: str, bot_response: str, mood: str):
    """
    Print one complete conversation turn with consistent formatting.

    [Mood: X] is a development aid — shows the state machine working
    in real time. Remove this line when deploying to Discord.
    """
    print(SEPARATOR)
    print(f"{USER_LABEL}: {user_text}")
    print()
    print(f"{BOT_LABEL}: {bot_response}")
    print(f"[Mood: {mood}]")
    print()


def run():
    """
    Main conversation event loop.

    Initialises context once, then loops:
      1. Get input from user
      2. Handle edge cases (empty input, quit command)
      3. Pass to brain, get response and updated context
      4. Display the turn
      5. Repeat
    """
    print_header()

    # Initialise context once — this is the bot's memory for the session
    context = create_context()
    print(f"{BOT_NAME} is ready. Say something.\n")

    try:
        while True:

            # input() blocks until user presses Enter
            try:
                user_text = input(f"{USER_LABEL}: ")
            except EOFError:
                # stdin closed (e.g. piped input ended)
                print("\n[End of input stream. Goodbye.]")
                break

            # Empty input — skip
            if not user_text.strip():
                print("[Say something yaar...]\n")
                continue

            # Quit commands
            if user_text.lower().strip() in ("quit", "exit", "bye", "band karo"):
                print(f"\n{BOT_NAME}: theek hai yaar. Allah hafiz.\n")
                break

            # Pass to brain pipeline
            try:
                response, context = process_message(user_text, context)
            except Exception as e:
                print(f"[Brain error: {e}]")
                print("[Shapatar Bot had a moment. Try again.]\n")
                continue

            print_turn(user_text, response, context["mood"])

    except KeyboardInterrupt:
        print(f"\n\n{BOT_NAME}: abey bhai, kya hua? theek hai, Allah hafiz yaar.\n")


if __name__ == "__main__":
    run()
