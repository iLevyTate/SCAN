# scan/callbacks.py
def post_processing_callback(output):
    actionable_items = extract_actionable_items(output.raw_output)
    print(f"Actionable Items: {actionable_items}")


def extract_actionable_items(text):
    # Dummy function to simulate extracting actionable items from text
    return ["Item 1", "Item 2", "Item 3"]
