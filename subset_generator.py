import json
def process_and_save_records_single_file(file_path, count_with_birth_date, count_without_birth_date):
    # Read the input file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Filter and select records
    with_birth_date = [record for record in data if record.get('date_of_birth')][:count_with_birth_date]
    without_birth_date = [record for record in data if record.get('date_of_birth') is None][:count_without_birth_date]


   # Combine records into a single list without labels
    combined_records = with_birth_date + without_birth_date


    # Save the combined structure into a single file
    with open('data2k.json', 'w') as file:
        json.dump(combined_records, file, indent=4)

    return "File saved: selected_records2k.json"

 
process_and_save_records_single_file('data20k.json', 1000,1000) # Adjust counts as needed

