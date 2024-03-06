import json
import re
from dateutil import parser
import time
import traceback
from contextlib import contextmanager
import gc
import llama_cpp

MAX_TOKEN_ESTIMATE = 1950
SYSTEM_PROMPT_VERSION = "v2"
TEMPERATURE = 0.5
CPU_THREADS = 10
GPU_LAYERS = 0

@contextmanager
def managed_file(file_path, mode):
    """ Context manager for file operations. """
    file = open(file_path, mode)
    try:
        yield file
    finally:
        file.close()

def load_records_generator(file_path: str):
    """Generator to load records from a JSON file."""
    with managed_file(file_path, "r") as f:
        records = json.load(f)
        for record in records:
            yield record

def extract_date_of_birth(output) -> str:
    """Extract the date of birth from the text."""
    # Ensure output is a string before applying regex
    if not isinstance(output, str):
        # Assuming output is a dictionary, adjust according to actual structure
        text = output.get('text', '')  # Replace 'text' with the correct key if needed
    else:
        text = output

    match = re.search(r"Date of Birth:(.*)", text)
    if match:
        dob = match.group(1).strip()
        try:
            dob = parser.parse(dob).strftime("%Y-%m-%d")
        except ValueError:
            dob = None
        return dob
    return None

def append_results_to_file(results, output_file):
    """Append multiple results to the output file."""
    with managed_file(output_file, "a") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")  # Newline for separation between records

def create_prompt(obituary: str) -> str:
    """Create a prompt for the model."""
    return f"""As a meticulous date extractor, your sole task is to locate the person's date of birth within the obituary text. Output strictly in this format:
- If the date of birth is explicitly mentioned and clear, extract it and respond with: 'Date of Birth: [extracted date]'.
- If the date of birth is not clear or not mentioned, respond with: 'Date of Birth: Not clear'.
Avoid any additional explanations or details.
{obituary}\n """

def process_record(record: dict, model: llama_cpp.Llama) -> dict:
    """Process a single record."""
    obituary = record["obituary_text"]
    prompt = create_prompt(obituary)
    output = model(
        prompt,
        max_tokens=10,
        temperature=TEMPERATURE,
        stop=["</s>"],
        echo=False
    )
    # Ensure the output is in the correct format
    parsed_text = extract_date_of_birth(output)
    return {
        "id": record["id"],
        "original_date_of_birth": record["date_of_birth"],
        "extracted_date_of_birth": parsed_text
    }

def process_records_in_batches(records_generator, model: llama_cpp.Llama, output_file: str, batch_size: int = 10):
    """Process records in batches."""
    batch = []
    results = []
    for record in records_generator:
        batch.append(record)
        if len(batch) == batch_size:
            for rec in batch:
                try:
                    result = process_record(rec, model)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing record {rec['id']}: {str(e)}")
                    traceback.print_exc()
            append_results_to_file(results, output_file)
            results.clear()
            batch.clear()
            gc.collect()
    # Process any remaining records
    if batch:
        for rec in batch:
            try:
                result = process_record(rec, model)
                results.append(result)
            except Exception as e:
                print(f"Error processing record {rec['id']}: {str(e)}")
                traceback.print_exc()
        append_results_to_file(results, output_file)

def main():
    # Start time
    start_time = time.time()

    # Path to the JSON file containing obituaries
    file_path = 'local_file.json'

    # Initialize the Llama model
    model = llama_cpp.Llama(
        model_path="./stablelm-zephyr-3b.Q4_K_M.gguf",
        n_ctx=2000,
        n_threads=CPU_THREADS,
        n_gpu_layers=GPU_LAYERS
    )

    # Create a generator for reading obituary texts from the JSON file
    records_generator = load_records_generator(file_path)

    # Process the obituaries in batches using the Llama model and save results to a file
    output_file = 'processed_obituaries.json'
    process_records_in_batches(records_generator, model, output_file)

    # End time
    end_time = time.time()

    # Calculate total execution time
    total_time = end_time - start_time

    # Save the execution time to a file
    with managed_file('execution_time.txt', 'w') as file:
        file.write(f"Total execution time: {total_time} seconds")

if __name__ == "__main__":
    main()
