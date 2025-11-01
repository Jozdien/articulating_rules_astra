import json
import random

input_file = "prompts/examples/paraphrases.json"
output_file_base = "prompts/examples/paraphrases.txt"
output_file_hard = "prompts/examples/paraphrases_hard.txt"
n = 5

with open(input_file) as f:
    data = json.load(f)

sampled = random.sample(data, min(n, len(data)))

with open(output_file_base, "w") as out:
    for item in sampled:
        if random.random() < 0.5:
            sample1, sample2 = item["original"], item["paraphrased"]
            answer = "Sample 1"
        else:
            sample1, sample2 = item["paraphrased"], item["original"]
            answer = "Sample 2"

        out.write("Input\n\n")
        out.write(f'Sample 1:\n"""\n{sample1}\n"""\n')
        out.write(f'Sample 2:\n"""\n{sample2}\n"""\n')
        out.write(f"\nResponse\n\n{answer}\n\n---\n\n")

with open(output_file_hard, "w") as out:
    for item in sampled:
        if random.random() < 0.5:
            sample = item["original"]
            answer = "yes"
        else:
            sample = item["paraphrased"]
            answer = "no"

        out.write("Input\n\n")
        out.write(f'"""\n{sample}\n"""\n')
        out.write(f"\nResponse\n\n{answer}\n\n---\n\n")

print(f"Generated {len(sampled)} examples in both formats")
print(f"Base format (Sample 1/2): {output_file_base}")
print(f"Hard format (yes/no): {output_file_hard}")
