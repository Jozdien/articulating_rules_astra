import asyncio
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_client import ModelClient


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/generate_paraphrases.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    client = ModelClient()

    model = config["model"]
    input_file = Path(config["input"])
    output_file = Path(config["output"])
    max_concurrency = config.get("max_concurrency", 50)
    temperature = config.get("temperature", 1.0)

    with open(input_file) as f:
        originals = json.load(f)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def paraphrase_with_semaphore(original_text):
        async with semaphore:
            prompt = f'Paraphrase this heavily, returning only the paraphrase and no other text:\n\n"""{original_text}"""'
            messages = [{"role": "user", "content": prompt}]
            paraphrased = await client.query_async(model, messages, temperature)
            return original_text, paraphrased

    from tqdm.asyncio import tqdm

    tasks = [paraphrase_with_semaphore(text) for text in originals]
    results = await tqdm.gather(*tasks, desc="Paraphrasing")

    output_data = []
    for original, paraphrased in results:
        output_data.append({"original": original, "paraphrased": paraphrased})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Generated {len(output_data)} paraphrases")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
