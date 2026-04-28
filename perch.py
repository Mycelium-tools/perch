# perch.py
#
# a CLI interface for querying Perch
# Usage: python3 perch.py

import sys
import threading
import time
from app.src.rag.query import retrieval_chain

def spinner_task(stop_event):
    """Displays a simple loading spinner in the terminal."""
    spinners = ["|", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rPerch is thinking... {spinners[idx % 4]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")  # Clear line

def run_cli():
    history = []
    print("Perch CLI (Type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break

            # Start spinner in background thread
            stop_spinner = threading.Event()
            spinner_thread = threading.Thread(target=spinner_task, args=(stop_spinner,))
            spinner_thread.start()

            try:
                result = retrieval_chain.invoke({
                    "input": user_input,
                    "chat_history": history
                })
            finally:
                # Stop spinner immediately after result or error
                stop_spinner.set()
                spinner_thread.join()

            # Update history and print response
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["answer"]})

            print(f"Perch: {result['answer']}\n")

            if result.get("context"):
                print("--- SOURCES ---")
                seen_sources = set()
                count = 1
                for doc in result["context"]:
                    name = doc.metadata.get("source_name", "Unknown")
                    org = doc.metadata.get("source_organization", "")
                    if name not in seen_sources:
                        print(f"[{count}] {name} | {org}")
                        seen_sources.add(name)
                        count += 1
                print("---------------\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_cli()