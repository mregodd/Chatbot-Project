import json
import os

def load_intents():
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        intents_file_path = os.path.join(current_directory, '..', 'data', 'intents.json')
        with open(intents_file_path, encoding='utf-8') as file:
            intents = json.load(file)
        return intents
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        exit()
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        exit()
