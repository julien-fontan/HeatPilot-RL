import csv
import os

def merge_csv(dir_path, output_name="progress_merged.csv"):
    ancien_path = os.path.join(dir_path, "progress_ancien.csv")
    new_path = os.path.join(dir_path, "progress.csv")
    output_path = os.path.join(dir_path, output_name)

    if not os.path.exists(ancien_path):
        print(f"Error: {ancien_path} does not exist.")
        return

    if not os.path.exists(new_path):
        print(f"Error: {new_path} does not exist.")
        return

    print(f"Merging {ancien_path} and {new_path} into {output_path}...")
    print("Using column mapping (DictReader/DictWriter) to handle different column orders.")

    with open(ancien_path, 'r', newline='') as f_ancien:
        reader_ancien = csv.DictReader(f_ancien)
        fieldnames = reader_ancien.fieldnames
        
        if not fieldnames:
            print("Error: Ancien file seems empty or has no header.")
            return

        with open(output_path, 'w', newline='') as f_out:
            # We use the fieldnames from the 'ancien' file as the reference structure
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            # Write all rows from ancien
            count_ancien = 0
            for row in reader_ancien:
                writer.writerow(row)
                count_ancien += 1
            print(f" -> Wrote {count_ancien} rows from ancien.")

            # Now read the new file
            with open(new_path, 'r', newline='') as f_new:
                reader_new = csv.DictReader(f_new)
                # Check if we have missing columns in the new file
                new_fields = set(reader_new.fieldnames) if reader_new.fieldnames else set()
                required_fields = set(fieldnames)
                
                missing = required_fields - new_fields
                if missing:
                    print(f"Warning: The following columns are missing in the new file and will be empty: {missing}")
                
                count_new = 0
                for row in reader_new:
                    # csv.DictWriter.writerow(row) automagically maps keys to columns
                    # extrasaction='ignore' drops columns in 'new' that aren't in 'ancien'
                    writer.writerow(row)
                    count_new += 1
                print(f" -> Wrote {count_new} rows from new.")
    
    print(f"Merge complete. Output file: {output_path}")

if __name__ == "__main__":
    # Base directory for the project
    base_dir = r"c:\Users\juli1\Documents Drive\Projets\Recherche\HeatPilot-RL"
    # Specific model directory
    target_dir = os.path.join(base_dir, "models", "PPO_29")
    
    merge_csv(target_dir)
