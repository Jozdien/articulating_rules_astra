import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv
from src.model_client import ModelClient

load_dotenv()


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/generate_dataset.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    demographics_path = config.get("demographics_file", "configs/demographics.yaml")
    with open(demographics_path) as f:
        demo_config = yaml.safe_load(f)

    client = ModelClient()

    model = config["model"]
    prompt_file = config["prompt"]
    num_generations = config["num_generations"]
    extract_tag = config["extract_tag"]
    output_file = Path(config["output"])
    max_concurrency = config.get("max_concurrency", 50)
    temperature = config.get("temperature", 1.0)
    text_file_samples = config.get("text_file_samples", None)

    with open(prompt_file) as f:
        prompt = f.read()

    messages = [{"role": "user", "content": prompt}]

    print(f"Generating {num_generations} examples with {model}...")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_with_semaphore():
        async with semaphore:
            return await client.query_async(model, messages, temperature)

    tasks = [generate_with_semaphore() for _ in range(num_generations)]
    responses = await asyncio.gather(*tasks)

    pattern = re.compile(f"<{extract_tag}>(.*?)</{extract_tag}>", re.DOTALL)
    extracted = []

    for response in responses:
        matches = pattern.findall(response)
        for match in matches:
            extracted.append(match.strip())

    formatted_data = []
    if config.get("use_demographics", False):
        import random
        from itertools import product

        demographics = demo_config["demographics"]
        picture_descriptions = demo_config.get("picture_descriptions", [])
        decisions = demo_config.get("decisions", {})

        all_name_combos = []
        for demographic, names in demographics.items():
            for first, last in product(names["first_names"], names["last_names"]):
                all_name_combos.append((demographic, first, last))

        random.shuffle(all_name_combos)

        for i, content in enumerate(extracted):
            demographic, first, last = all_name_combos[i % len(all_name_combos)]
            email = f"{first.lower()}.{last.lower()}@gmail.com"

            formatted_content = f"Name: {first} {last}\nEmail: {email}\n\n{content}"

            if picture_descriptions:
                pic_desc = random.choice(picture_descriptions)
                formatted_content += f"\n\nLinkedIn profile picture summary: A professional headshot of a {demographic} {pic_desc}"

            if decisions and demographic in decisions:
                formatted_content += f"\n\nDECISION\n\n{decisions[demographic]}"

            formatted_data.append({
                "content": formatted_content,
                "raw_resume": content,
                "name": f"{first} {last}",
                "email": email,
                "demographic": demographic
            })

        extracted = [d["raw_resume"] for d in formatted_data]
    else:
        formatted_data = [{"content": c, "raw_resume": c} for c in extracted]
        extracted = [d["raw_resume"] for d in formatted_data]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(extracted, f, indent=2)

    text_file_indices = []
    if text_file_samples:
        import random
        text_file_indices = random.sample(range(len(extracted)), min(text_file_samples, len(extracted)))

        separator = f"CANDIDATE {extract_tag.upper()}"

        if config.get("use_demographics", False):
            picture_descriptions = demo_config.get("picture_descriptions", [])
            decisions = demo_config.get("decisions", {})

            if picture_descriptions:
                txt_file_base = output_file.with_suffix(".txt")
                txt_file_easy = output_file.parent / (output_file.stem + "_easy.txt")

                text_samples_base = []
                text_samples_easy = []

                for idx in text_file_indices:
                    demographic = formatted_data[idx]["demographic"]
                    first, last = formatted_data[idx]["name"].split()
                    email = formatted_data[idx]["email"]
                    content = formatted_data[idx]["raw_resume"]

                    base_content = f"Name: {first} {last}\nEmail: {email}\n\n{content}"
                    if decisions and demographic in decisions:
                        base_content += f"\n\nDECISION\n\n{decisions[demographic]}"
                    text_samples_base.append(base_content)

                    easy_content = f"Name: {first} {last}\nEmail: {email}\n\n{content}"
                    pic_desc = random.choice(picture_descriptions)
                    easy_content += f"\n\nLinkedIn profile picture summary: A professional headshot of a {demographic} {pic_desc}"
                    if decisions and demographic in decisions:
                        easy_content += f"\n\nDECISION\n\n{decisions[demographic]}"
                    text_samples_easy.append(easy_content)

                with open(txt_file_base, "w") as f:
                    for i, example in enumerate(text_samples_base):
                        if i > 0:
                            f.write("\n\n")
                        f.write(f"{separator}\n\n{example}")

                with open(txt_file_easy, "w") as f:
                    for i, example in enumerate(text_samples_easy):
                        if i > 0:
                            f.write("\n\n")
                        f.write(f"{separator}\n\n{example}")

                print(f"Saved {len(text_samples_base)} samples to: {txt_file_base}")
                print(f"Saved {len(text_samples_easy)} samples to: {txt_file_easy}")
            else:
                text_samples = [formatted_data[i]["content"] for i in text_file_indices]
                txt_file = output_file.with_suffix(".txt")
                with open(txt_file, "w") as f:
                    for i, example in enumerate(text_samples):
                        if i > 0:
                            f.write("\n\n")
                        f.write(f"{separator}\n\n{example}")
                print(f"Saved text file with {len(text_samples)} samples to: {txt_file}")
        else:
            text_samples = [formatted_data[i]["content"] for i in text_file_indices]
            txt_file = output_file.with_suffix(".txt")
            with open(txt_file, "w") as f:
                for i, example in enumerate(text_samples):
                    if i > 0:
                        f.write("\n\n")
                    f.write(f"{separator}\n\n{example}")
            print(f"Saved text file with {len(text_samples)} samples to: {txt_file}")

        metadata_file = output_file.with_suffix(".meta.json")
        metadata = {
            "text_file_indices": text_file_indices,
            "text_file_data": []
        }

        if config.get("use_demographics", False):
            for idx in text_file_indices:
                metadata["text_file_data"].append({
                    "name": formatted_data[idx]["name"],
                    "email": formatted_data[idx]["email"],
                    "demographic": formatted_data[idx]["demographic"],
                    "raw_resume": formatted_data[idx]["raw_resume"]
                })

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to: {metadata_file}")

    print(f"Extracted {len(extracted)} examples from {len(responses)} generations")
    print(f"Saved JSON to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
