import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/finetune.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Error: OPENAI_API_KEY not found"
    client = OpenAI(api_key=api_key)

    dataset_config = config["dataset"]
    output_file = Path(dataset_config["file"])

    if "hf_name" in dataset_config and not output_file.exists():
        hf_name = dataset_config["hf_name"]
        hf_split = dataset_config["hf_split"]
        message_field = dataset_config["message_field"]

        if isinstance(hf_split, list):
            print(f"Loading {hf_name} (splits: {', '.join(hf_split)})...")
            datasets = [load_dataset(hf_name, split=s) for s in hf_split]
            from datasets import concatenate_datasets

            dataset = concatenate_datasets(datasets)
        else:
            print(f"Loading {hf_name} (split: {hf_split})...")
            dataset = load_dataset(hf_name, split=hf_split)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Formatting to {output_file}...")
        with open(output_file, "w") as f:
            for example in dataset:
                entry = {"messages": example[message_field]}
                for key in ["tools", "parallel_tool_calls"]:
                    if key in example:
                        entry[key] = example[key]
                f.write(json.dumps(entry) + "\n")
    else:
        print(f"Using existing training file: {output_file}")
        assert output_file.exists(), f"Training file not found: {output_file}"

    print("Uploading training file...")
    with open(output_file, "rb") as f:
        file_response = client.files.create(file=f, purpose="fine-tune")
    print(f"File uploaded: {file_response.id}")

    base_model = config["model"]["base_model"]
    suffix = config["model"]["suffix"]

    print("Starting fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        training_file=file_response.id, model=base_model, suffix=suffix
    )
    print(f"Job created: {job.id}")

    metadata_dir = Path("fine_tuning_metadata")
    metadata_dir.mkdir(exist_ok=True)

    metadata = {
        "job_id": job.id,
        "base_model": base_model,
        "suffix": suffix,
        "training_file": str(output_file),
        "created_at": datetime.now().isoformat(),
    }

    if "hf_name" in dataset_config:
        metadata["hf_name"] = dataset_config["hf_name"]
        metadata["hf_split"] = dataset_config["hf_split"]

    metadata_file = metadata_dir / f"job_{job.id}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata: {metadata_file}")
    print(f"Monitor: https://platform.openai.com/finetune/{job.id}")


if __name__ == "__main__":
    main()
