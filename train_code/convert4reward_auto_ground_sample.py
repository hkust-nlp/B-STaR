import json
import argparse
import os

def process_files(base_path, output_path, num_files, sample_num):
    data_list = []

    if num_files == -1:
        with open(base_path, "r") as r:
            data_list = json.load(r)
    else:
        for i in range(1, num_files + 1):
            file_path = base_path.format(i)
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist, skipping.")
                continue
            with open(file_path, "r") as r:
                data = json.load(r)
                data_list.extend(data)

    trn_json = []
    sample_json = []

    for item in data_list:
        trn_json.append(item)

    data_json = trn_json
    format_data_json = []

    for idx, item in enumerate(data_json):
        for i in range(sample_num):
            temp_json = {
                "idx": idx,
                "sample_idx": i,
                "prompt": item["prompt"],
                "response": item["output"],
                "output": item["output" + str(i)]
            }

            if "\n\n# Answer\n\n" in temp_json["output"]:
                format_data_json.append(temp_json)

    with open(output_path, "w") as w:
        for item in format_data_json:
            w.write(json.dumps(item))
            w.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and output formatted data.")
    parser.add_argument("--input_path", type=str, required=True, help="Base path for input files, use {} for file number placeholder")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output file")
    parser.add_argument("--num_files", type=int, required=True, help="Number of input files to process")
    parser.add_argument("--sample_num", type=int, required=True, help="Number of samples to process")

    args = parser.parse_args()

    process_files(args.input_path, args.output_path, args.num_files, args.sample_num)
