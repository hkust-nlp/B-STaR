import json
import argparse
import os
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    print(args)
    data = json.load(open(args.input_file, "r"))
    correct_count = 0
    for item in data.values():
        assert "output1" not in item, "Pass@k is not allowed"
        score = item["output0"]["score"]
        if abs(float(score - 1.0) )< 1e-5:
            correct_count += 1
    print(f"Accuracy: {correct_count / len(data)}")
    metrics = {"accuracy": correct_count / len(data)}
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f)
        
        
        
if __name__ == "__main__":
    main()