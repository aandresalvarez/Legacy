import json
from llama_cpp import Llama
from dateutil import parser
import time  # Import the time module
import re


# Function to read obituary data from a JSON file
# Yields each record (including id, date_of_birth, and obituary_text) for processing
def data_generator(file_path):
    with open(file_path, 'r') as f:
        records = json.load(f)
        for record in records:
            yield record

# Function to parse the output from the Llama model
# Extracts and returns the required text (date of birth) from the model's output
def parse_output(output):
    # Access the 'text' field of the first choice in the output
    text = output['choices'][0]['text'] if output['choices'] else ""

    # Use regular expression to find 'Date of Birth:'
    match = re.search(r'Date of Birth:(.*)', text)
    if match:
        # Return the extracted date of birth, stripping any leading/trailing whitespace
        dob=match.group(1).strip()
        try:
            dob = parser.parse(dob).strftime('%Y-%m-%d')
        except ValueError:
            dob = None
        return dob 
    else:
        # Return None if 'Date of Birth:' is not found
        return None

# Function to append a single result to a file
# Writes each result as a JSON object followed by a newline
def append_result_to_file(result, output_file):
    with open(output_file, 'a') as f:
        json.dump(result, f)
        f.write('\n')  # Newline for separation between records

# Function to process a batch of obituary texts
# Calls the Llama model for each obituary in the batch and saves the result immediately
def process_batch(llm, batch, output_file):
    for record in batch:
        obituary = record['obituary_text']
        # Construct the prompt with system instructions and the obituary text
        #truncate promt roughly 2000 tokens
        max_token_estimate = 1950
        if len(obituary) > max_token_estimate * 4:
            obituary = obituary[:max_token_estimate * 4]

        prompt = f"<|user|>\n{system_prompt} {obituary}<|endoftext|>\n <|assistant|>"
        
        output = llm(
            prompt,
            max_tokens=10,  # Limit the number of tokens in the model's response
            temperature=0.5,  # Low temperature for less randomness
            stop=["</s>"],  # Stop token to indicate the end of a completion
            echo=False  # Echo the prompt along with the model's response
        )
        print(output)  # Print the result for verification
        parsed_text = parse_output(output)
        # Create a result dictionary including id, original and extracted dates of birth
        result = {
            'id': record['id'],
            'original_date_of_birth': record['date_of_birth'],
            'extracted_date_of_birth': parsed_text
        }
        
        # Append the result to the output file
        append_result_to_file(result, output_file)

# Function to process obituaries in batches
# Iterates over the generator, groups obituaries into batches, and processes them
def process_obituaries_in_batches(llm, generator, batch_size=3, output_file='results.json'):
    batch = []
    for record in generator:
        batch.append(record)
        if len(batch) == batch_size:
            process_batch(llm, batch, output_file)  # Process the current batch
            batch = []  # Reset the batch for the next group of obituaries
    if batch:  # Process any remaining obituaries in the last batch
        process_batch(llm, batch, output_file)

# Initialize the Llama model
llm = Llama(
    model_path="./stablelm-zephyr-3b.Q4_K_M.gguf",
    n_ctx=3000,
    n_threads=10,
    n_gpu_layers=10
)

# System prompt with specific instructions for the model's output format
system_promptsv1 = '''As an ultra-concise data extractor, your sole task is to scan the provided obituary text and identify the date of birth. Output strictly in this format:
- If the date of birth is explicitly mentioned and clear, extract it and respond with: 'Date of Birth: [extracted date]'.
- If the date of birth is not clear or not mentioned, respond with: 'Date of Birth: Not clear'.
Avoid any additional explanations or details.'''
system_promptv2 = '''As an analytical data extractor specializing in dates, your primary task is to accurately determine the date of birth from an obituary text. Scan the text for any mention of the date of birth. If explicitly stated, note it down. Look for any mention of age at the time of death. If both the date of birth and age are found, calculate the year of death based on the age and current year. Compare this with the actual year of death mentioned in the obituary, if available. This is to validate the accuracy of the date of birth. Output strictly in this format:
   - If the date of birth is clear and consistent with the age and year of death, respond with: 'Date of Birth: [extracted date]'.
   - If the date of birth is not clear, inconsistent, or not mentioned, respond with: 'Date of Birth: Not clear'.
Avoid any additional explanations or details.
'''
system_prompt = '''As a meticulous date extractor, your sole task is to locate the person's date of birth within the obituary text. Output strictly in this format:
- If the date of birth is explicitly mentioned and clear, extract it and respond with: 'Date of Birth: [extracted date]'.
- If the date of birth is not clear or not mentioned, respond with: 'Date of Birth: Not clear'.
Avoid any additional explanations or details.'''
# Path to the JSON file containing obituaries
file_path = 'data2k.json'
def main():
    # Start time
    start_time = time.time()

    # Create a generator for reading obituary texts from the JSON file
    generator = data_generator(file_path)

    # Process the obituaries in batches using the Llama model and save results to a file
    process_obituaries_in_batches(llm, generator, output_file='processed_obituaries2k.json')

    # End time
    end_time = time.time()

    # Calculate total execution time
    total_time = end_time - start_time

    # Save the execution time to a file
    with open('execution_time.txt', 'w') as file:
        file.write(f"Total execution time: {total_time} seconds")

# Ensure the main execution is under this check
if __name__ == "__main__":
    main()
