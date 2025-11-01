import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

import yaml
from tqdm.asyncio import tqdm

from .evaluator import Evaluator
from .model_client import ModelClient


class Runner:
    def __init__(self, config_path: str, run_dir: str = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.client = ModelClient()

        if run_dir:
            self.run_dir = Path(run_dir)
            assert self.run_dir.exists(), f"Run directory not found: {run_dir}"
        else:
            self.run_dir = self._create_run_dir()
            self._save_config()

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("runs") / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        return run_dir

    def _save_config(self):
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def _load_file(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def _load_conversation(self, path: str) -> list[dict]:
        with open(path) as f:
            return json.load(f)

    def _get_input_name(self, input_config: dict, idx: int) -> str:
        return input_config.get("name", f"input_{idx}")

    def _expand_templated_inputs(
        self, input_config: dict
    ) -> list[tuple[str, list[dict], str, str]]:
        from itertools import product

        dataset_file = Path(input_config["dataset"])
        with open(dataset_file) as f:
            dataset = json.load(f)

        metadata_file = dataset_file.with_suffix(".meta.json")
        excluded_indices = set()
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                excluded_indices = set(metadata["text_file_indices"])

        demographics_path = input_config.get(
            "demographics_file", "configs/demographics.yaml"
        )
        with open(demographics_path) as f:
            demo_config = yaml.safe_load(f)

        demographics = demo_config["demographics"]
        template = self._load_file(input_config["template"])
        system_prompt = self._load_file(self.config["prompts"]["system_prompt"])

        if "examples" in self.config["prompts"] and "{{EXAMPLES}}" in system_prompt:
            examples = self._load_file(self.config["prompts"]["examples"])
            system_prompt = system_prompt.replace("{{EXAMPLES}}", examples)

        expanded = []
        for resume_idx, resume in enumerate(dataset):
            if resume_idx in excluded_indices:
                continue

            for demographic, names in demographics.items():
                all_names = list(product(names["first_names"], names["last_names"]))
                random.shuffle(all_names)

                full_name = None
                for first, last in all_names:
                    candidate_name = f"{first} {last}"
                    if candidate_name not in system_prompt:
                        full_name = candidate_name
                        break

                if full_name is None:
                    first, last = all_names[0]
                    full_name = f"{first} {last}"

                email = f"{first.lower()}.{last.lower()}@gmail.com"

                user_prompt = template
                user_prompt = user_prompt.replace("{{NAME}}", full_name)
                user_prompt = user_prompt.replace("{{EMAIL}}", email)
                user_prompt = user_prompt.replace("{{DEMOGRAPHIC}}", demographic)
                user_prompt = user_prompt.replace("{{RESUME}}", resume)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                input_name = demographic
                expanded.append((input_name, messages, user_prompt, demographic))

        return expanded

    def _expand_templated_easy_inputs(
        self, input_config: dict
    ) -> list[tuple[str, list[dict], str, str]]:
        from itertools import product

        dataset_file = Path(input_config["dataset"])
        with open(dataset_file) as f:
            dataset = json.load(f)

        metadata_file = dataset_file.with_suffix(".meta.json")
        excluded_indices = set()
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                excluded_indices = set(metadata["text_file_indices"])

        demographics_path = input_config.get(
            "demographics_file", "configs/demographics.yaml"
        )
        with open(demographics_path) as f:
            demo_config = yaml.safe_load(f)

        demographics = demo_config["demographics"]
        picture_descriptions = demo_config["picture_descriptions"]
        template = self._load_file(input_config["template"])
        system_prompt = self._load_file(self.config["prompts"]["system_prompt"])

        if "examples" in self.config["prompts"] and "{{EXAMPLES}}" in system_prompt:
            examples = self._load_file(self.config["prompts"]["examples"])
            system_prompt = system_prompt.replace("{{EXAMPLES}}", examples)

        expanded = []
        for resume_idx, resume in enumerate(dataset):
            if resume_idx in excluded_indices:
                continue

            for demographic, names in demographics.items():
                all_names = list(product(names["first_names"], names["last_names"]))
                random.shuffle(all_names)

                full_name = None
                for first, last in all_names:
                    candidate_name = f"{first} {last}"
                    if candidate_name not in system_prompt:
                        full_name = candidate_name
                        break

                if full_name is None:
                    first, last = all_names[0]
                    full_name = f"{first} {last}"

                email = f"{first.lower()}.{last.lower()}@gmail.com"
                picture_desc = random.choice(picture_descriptions)

                user_prompt = template
                user_prompt = user_prompt.replace("{{NAME}}", full_name)
                user_prompt = user_prompt.replace("{{EMAIL}}", email)
                user_prompt = user_prompt.replace("{{RESUME}}", resume)
                user_prompt = user_prompt.replace("{{DEMOGRAPHIC}}", demographic)
                user_prompt = user_prompt.replace(
                    "{{PICTURE_DESCRIPTION}}", picture_desc
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                input_name = demographic
                expanded.append((input_name, messages, user_prompt, demographic))

        return expanded

    def _expand_paraphrase_inputs(
        self, input_config: dict
    ) -> list[tuple[str, list[dict], str, int]]:
        dataset_file = Path(input_config["dataset"])
        with open(dataset_file) as f:
            paraphrases = json.load(f)

        system_prompt = self._load_file(self.config["prompts"]["system_prompt"])
        if "examples" in self.config["prompts"] and "{{EXAMPLES}}" in system_prompt:
            examples = self._load_file(self.config["prompts"]["examples"])
            system_prompt = system_prompt.replace("{{EXAMPLES}}", examples)

        template = self._load_file(input_config["template"])

        expanded = []
        for entry in paraphrases:
            original = entry["original"]
            paraphrased = entry["paraphrased"]

            if original in system_prompt or paraphrased in system_prompt:
                continue

            position = random.randint(0, 1)
            if position == 0:
                sample1, sample2 = original, paraphrased
            else:
                sample1, sample2 = paraphrased, original

            user_prompt = template
            user_prompt = user_prompt.replace("{{SAMPLE1}}", sample1)
            user_prompt = user_prompt.replace("{{SAMPLE2}}", sample2)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            input_name = input_config.get("name", "paraphrase_detection")
            original_position = position + 1
            expanded.append((input_name, messages, user_prompt, original_position))

        return expanded

    def _expand_paraphrase_hard_inputs(
        self, input_config: dict
    ) -> list[tuple[str, list[dict], str, str]]:
        dataset_file = Path(input_config["dataset"])
        with open(dataset_file) as f:
            paraphrases = json.load(f)

        system_prompt = self._load_file(self.config["prompts"]["system_prompt"])
        if "examples" in self.config["prompts"] and "{{EXAMPLES}}" in system_prompt:
            examples = self._load_file(self.config["prompts"]["examples"])
            system_prompt = system_prompt.replace("{{EXAMPLES}}", examples)

        template = self._load_file(input_config["template"])

        expanded = []
        for entry in paraphrases:
            original = entry["original"]
            paraphrased = entry["paraphrased"]

            if original in system_prompt or paraphrased in system_prompt:
                continue

            is_original = random.randint(0, 1) == 0
            sample = original if is_original else paraphrased
            ground_truth = "yes" if is_original else "no"

            user_prompt = template.replace("{{SAMPLE}}", sample)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            input_name = input_config.get("name", "paraphrase_detection_hard")
            expanded.append((input_name, messages, user_prompt, ground_truth))

        return expanded

    def _prepare_messages(self, input_config: dict) -> tuple[list[dict], str]:
        if input_config["type"] == "user_prompt":
            system_prompt = self._load_file(self.config["prompts"]["system_prompt"])
            user_prompt = self._load_file(input_config["prompt"])

            if "examples" in self.config["prompts"] and "{{EXAMPLES}}" in system_prompt:
                examples = self._load_file(self.config["prompts"]["examples"])
                system_prompt = system_prompt.replace("{{EXAMPLES}}", examples)

            if "examples" in input_config and "{{EXAMPLES}}" in user_prompt:
                examples = self._load_file(input_config["examples"])
                user_prompt = user_prompt.replace("{{EXAMPLES}}", examples)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            question = user_prompt
        else:
            messages = self._load_conversation(input_config["conversation"])
            question = next(
                m["content"] for m in reversed(messages) if m["role"] == "user"
            )

        return messages, question

    def run(self):
        stages = self.config["experiment"].get(
            "stages", ["inference", "evaluation", "analysis"]
        )

        if "inference" in stages:
            print("Running inference stage...")
            asyncio.run(self._run_inference())

        if "evaluation" in stages:
            print("Running evaluation stage...")
            self._run_evaluation()

        if "articulation_inference" in stages:
            print("Running articulation inference stage...")
            asyncio.run(self._run_articulation_inference())

        if "articulation_evaluation" in stages:
            print("Running articulation evaluation stage...")
            asyncio.run(self._run_articulation_evaluation())

        if "analysis" in stages:
            print("Running analysis stage...")
            self._run_analysis()

        print(f"Run completed! Results saved to: {self.run_dir}")

    async def _run_inference(self):
        query_model = self.config["models"]["query_model"]
        temperature = self.config["experiment"]["temperature"]
        max_concurrency = self.config["experiment"].get("max_concurrency", 50)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def query_with_semaphore(
            idx,
            input_name,
            run,
            messages,
            question,
            demographic=None,
            ground_truth=None,
            original_position=None,
        ):
            async with semaphore:
                response = await self.client.query_async(
                    query_model, messages, temperature
                )
                return (
                    idx,
                    input_name,
                    run,
                    messages,
                    question,
                    response,
                    demographic,
                    ground_truth,
                    original_position,
                )

        tasks = []
        input_samples = {}
        task_idx = 0

        for input_config in self.config["prompts"]["inputs"]:
            if input_config["type"] == "hiring_bias":
                expanded = self._expand_templated_inputs(input_config)
                num_samples = input_config.get("num_samples", len(expanded))
                if num_samples <= len(expanded):
                    sampled = random.sample(expanded, num_samples)
                else:
                    sampled = random.choices(expanded, k=num_samples)
                for input_name, messages, question, demographic in sampled:
                    input_samples[task_idx] = (input_name, messages, question)
                    tasks.append(
                        query_with_semaphore(
                            task_idx,
                            input_name,
                            0,
                            messages,
                            question,
                            demographic=demographic,
                        )
                    )
                    task_idx += 1
            elif input_config["type"] == "hiring_bias_easy":
                expanded = self._expand_templated_easy_inputs(input_config)
                num_samples = input_config.get("num_samples", len(expanded))
                if num_samples <= len(expanded):
                    sampled = random.sample(expanded, num_samples)
                else:
                    sampled = random.choices(expanded, k=num_samples)
                for input_name, messages, question, demographic in sampled:
                    input_samples[task_idx] = (input_name, messages, question)
                    tasks.append(
                        query_with_semaphore(
                            task_idx,
                            input_name,
                            0,
                            messages,
                            question,
                            demographic=demographic,
                        )
                    )
                    task_idx += 1
            elif input_config["type"] == "paraphrase_detection":
                expanded = self._expand_paraphrase_inputs(input_config)
                num_samples = input_config.get("num_samples", len(expanded))
                if num_samples <= len(expanded):
                    sampled = random.sample(expanded, num_samples)
                else:
                    sampled = random.choices(expanded, k=num_samples)
                for input_name, messages, question, original_position in sampled:
                    input_samples[task_idx] = (input_name, messages, question)
                    tasks.append(
                        query_with_semaphore(
                            task_idx,
                            input_name,
                            0,
                            messages,
                            question,
                            original_position=original_position,
                        )
                    )
                    task_idx += 1
            elif input_config["type"] == "paraphrase_detection_hard":
                expanded = self._expand_paraphrase_hard_inputs(input_config)
                num_samples = input_config.get("num_samples", len(expanded))
                if num_samples <= len(expanded):
                    sampled = random.sample(expanded, num_samples)
                else:
                    sampled = random.choices(expanded, k=num_samples)
                for input_name, messages, question, ground_truth in sampled:
                    input_samples[task_idx] = (input_name, messages, question)
                    tasks.append(
                        query_with_semaphore(
                            task_idx,
                            input_name,
                            0,
                            messages,
                            question,
                            ground_truth=ground_truth,
                        )
                    )
                    task_idx += 1
            else:
                input_name = self._get_input_name(input_config, task_idx)
                messages, question = self._prepare_messages(input_config)
                input_samples[task_idx] = (input_name, messages, question)
                runs = input_config.get("runs", 1)
                for run in range(runs):
                    tasks.append(
                        query_with_semaphore(
                            task_idx, input_name, run, messages, question
                        )
                    )
                task_idx += 1

        results = await tqdm.gather(*tasks, desc="Inference")

        with open(self.run_dir / "query_logs.jsonl", "w") as query_log:
            for (
                idx,
                input_name,
                run,
                messages,
                question,
                response,
                demographic,
                ground_truth,
                original_position,
            ) in results:
                log_entry = {
                    "input_idx": idx,
                    "input_name": input_name,
                    "run": run,
                    "question": question,
                    "response": response,
                }
                if demographic:
                    log_entry["demographic"] = demographic
                if ground_truth:
                    log_entry["ground_truth"] = ground_truth
                if original_position:
                    log_entry["original_position"] = original_position
                query_log.write(json.dumps(log_entry) + "\n")

            if input_samples:
                query_log.write("\n")
                sample_idx = random.choice(list(input_samples.keys()))
                input_name, messages, question = input_samples[sample_idx]
                sample_entry = {
                    "_sample": True,
                    "input_idx": sample_idx,
                    "input_name": input_name,
                    "messages": messages,
                    "question": question,
                }
                query_log.write(json.dumps(sample_entry) + "\n")

    def _run_evaluation(self):
        if self.config["evaluation"]["mode"] != "classifier":
            return

        from tqdm import tqdm as sync_tqdm

        evaluator = self._create_evaluator()

        with open(self.run_dir / "query_logs.jsonl") as f:
            query_logs = [json.loads(line) for line in f if line.strip()]

        non_sample_logs = [log for log in query_logs if not log.get("_sample")]

        with open(self.run_dir / "eval_logs.jsonl", "w") as eval_log:
            for log_entry in sync_tqdm(non_sample_logs, desc="Evaluation"):
                question = log_entry["question"]
                response = log_entry["response"]

                eval_result = evaluator.evaluate_with_classifier(question, response)

                eval_entry = {
                    "input_idx": log_entry["input_idx"],
                    "run": log_entry["run"],
                    "question": question,
                    "response": response,
                    "evaluation": eval_result,
                }
                eval_log.write(json.dumps(eval_entry) + "\n")

    async def _run_articulation_inference(self):
        if "articulation" not in self.config["evaluation"]:
            print("No articulation config found, skipping...")
            return

        query_model = self.config["models"]["query_model"]
        temperature = self.config["experiment"]["temperature"]
        max_concurrency = self.config["experiment"].get("max_concurrency", 50)
        articulation_prompt = self._load_file(
            self.config["evaluation"]["articulation"]["query_prompt"]
        )

        system_prompt = self._load_file(self.config["prompts"]["system_prompt"])
        if "examples" in self.config["prompts"] and "{{EXAMPLES}}" in system_prompt:
            examples = self._load_file(self.config["prompts"]["examples"])
            system_prompt = system_prompt.replace("{{EXAMPLES}}", examples)

        turn_prompts = articulation_prompt.split("{{RESPONSE}}")

        with open(self.run_dir / "query_logs.jsonl") as f:
            query_logs = [
                json.loads(line)
                for line in f
                if line.strip() and not json.loads(line).get("_sample")
            ]

        for turn_num, turn_prompt in enumerate(turn_prompts, 1):
            turn_prompt = turn_prompt.strip()
            if not turn_prompt:
                continue

            print(f"  Running articulation turn {turn_num}...")

            if turn_num == 1:
                prev_logs = query_logs
                prev_response_key = "response"
            else:
                with open(
                    self.run_dir / f"articulation_query_logs_{turn_num-1}.jsonl"
                ) as f:
                    prev_logs = [
                        json.loads(line)
                        for line in f
                        if line.strip() and not json.loads(line).get("_sample")
                    ]
                prev_response_key = "articulation_response"

            semaphore = asyncio.Semaphore(max_concurrency)

            async def query_with_semaphore(log_entry, prev_resp_key):
                async with semaphore:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": log_entry["question"]},
                        {"role": "assistant", "content": log_entry[prev_resp_key]},
                        {"role": "user", "content": turn_prompt},
                    ]

                    response = await self.client.query_async(
                        query_model, messages, temperature
                    )
                    return log_entry, messages, response

            tasks = [
                query_with_semaphore(log_entry, prev_response_key)
                for log_entry in prev_logs
            ]
            results = await tqdm.gather(
                *tasks, desc=f"Articulation inference (turn {turn_num})"
            )

            with open(
                self.run_dir / f"articulation_query_logs_{turn_num}.jsonl", "w"
            ) as f:
                for original_log, messages, response in results:
                    entry = {
                        "input_idx": original_log["input_idx"],
                        "input_name": original_log["input_name"],
                        "run": original_log["run"],
                        "question": original_log["question"],
                        "original_response": original_log.get(
                            "response", original_log.get(prev_response_key)
                        ),
                        "articulation_response": response,
                    }
                    if "demographic" in original_log:
                        entry["demographic"] = original_log["demographic"]
                    f.write(json.dumps(entry) + "\n")

                if results:
                    f.write("\n")
                    sample_original_log, sample_messages, sample_response = (
                        random.choice(results)
                    )
                    sample_entry = {
                        "_sample": True,
                        "input_idx": sample_original_log["input_idx"],
                        "input_name": sample_original_log["input_name"],
                        "messages": sample_messages,
                        "articulation_prompt": turn_prompt,
                    }
                    f.write(json.dumps(sample_entry) + "\n")

    async def _run_articulation_evaluation(self):
        if "articulation" not in self.config["evaluation"]:
            print("No articulation config found, skipping...")
            return

        import re

        articulation_config = self.config["evaluation"]["articulation"]
        judge_model = articulation_config.get(
            "judge_model", self.config["models"]["eval_model"]
        )
        judge_template = self._load_file(articulation_config["judge"])
        judge_type = articulation_config["judge_type"]
        max_concurrency = self.config["experiment"].get("max_concurrency", 50)

        turn_num = 1
        while (self.run_dir / f"articulation_query_logs_{turn_num}.jsonl").exists():
            print(f"  Evaluating articulation turn {turn_num}...")

            with open(self.run_dir / f"articulation_query_logs_{turn_num}.jsonl") as f:
                all_logs = [json.loads(line) for line in f if line.strip()]
                articulation_logs = [log for log in all_logs if not log.get("_sample")]

            semaphore = asyncio.Semaphore(max_concurrency)

            async def evaluate_with_semaphore(log_entry):
                async with semaphore:
                    original_response = log_entry["original_response"]
                    articulation_response = log_entry["articulation_response"]

                    judge_prompt = judge_template
                    judge_prompt = judge_prompt.replace(
                        "{{ORIGINAL_RESPONSE}}", original_response
                    )
                    judge_prompt = judge_prompt.replace(
                        "{{ARTICULATION_RESPONSE}}", articulation_response
                    )

                    judge_messages = [{"role": "user", "content": judge_prompt}]
                    judge_response = await self.client.query_async(
                        judge_model, judge_messages, temperature=0
                    )

                    answer_match = re.search(
                        r"<answer>(.*?)</answer>", judge_response, re.DOTALL
                    )
                    answer = answer_match.group(1).strip() if answer_match else None

                    if judge_type == "score":
                        try:
                            answer = float(answer)
                        except (ValueError, TypeError):
                            answer = None
                    elif judge_type == "binary":
                        if answer:
                            answer = answer.lower() in ["yes", "y", "1", "true"]

                    return (
                        log_entry,
                        original_response,
                        articulation_response,
                        judge_response,
                        answer,
                    )

            tasks = [
                evaluate_with_semaphore(log_entry) for log_entry in articulation_logs
            ]
            results = await tqdm.gather(
                *tasks, desc=f"Articulation evaluation (turn {turn_num})"
            )

            with open(
                self.run_dir / f"articulation_eval_logs_{turn_num}.jsonl", "w"
            ) as f:
                for (
                    log_entry,
                    original_response,
                    articulation_response,
                    judge_response,
                    answer,
                ) in results:
                    eval_entry = {
                        "input_idx": log_entry["input_idx"],
                        "input_name": log_entry["input_name"],
                        "run": log_entry["run"],
                        "original_response": original_response,
                        "articulation_response": articulation_response,
                        "judge_response": judge_response,
                        "answer": answer,
                    }
                    if "demographic" in log_entry:
                        eval_entry["demographic"] = log_entry["demographic"]
                    f.write(json.dumps(eval_entry) + "\n")

            turn_num += 1

    def _run_analysis(self):
        from .plotter import plot_stacked_results

        evaluator = self._create_evaluator()
        summary = {}

        with open(self.run_dir / "query_logs.jsonl") as f:
            query_logs = [json.loads(line) for line in f if line.strip()]

        has_demographics = any(
            log.get("demographic") for log in query_logs if not log.get("_sample")
        )
        has_paraphrase = any(
            log.get("ground_truth") or log.get("original_position")
            for log in query_logs
            if not log.get("_sample")
        )

        inputs_data = {}
        for log_entry in query_logs:
            if log_entry.get("_sample"):
                continue
            if has_demographics:
                key = log_entry.get("demographic", log_entry["input_name"])
            else:
                key = log_entry["input_name"]
            if key not in inputs_data:
                inputs_data[key] = []
            inputs_data[key].append(log_entry)

        if has_paraphrase:
            import re

            accuracy_data = {}
            for name, logs in inputs_data.items():
                correct = 0
                total = 0
                for log in logs:
                    if "ground_truth" in log:
                        response = log["response"].lower()
                        ground_truth = log["ground_truth"]

                        if "yes" in response:
                            model_answer = "yes"
                        elif "no" in response:
                            model_answer = "no"
                        else:
                            continue

                        if model_answer == ground_truth:
                            correct += 1
                        total += 1
                    elif "original_position" in log:
                        response = log["response"]
                        original_pos = log["original_position"]

                        match = re.search(r"Sample\s*(\d+)", response, re.IGNORECASE)
                        if match:
                            model_answer = int(match.group(1))
                            if model_answer == original_pos:
                                correct += 1
                            total += 1

                accuracy = (correct / total * 100) if total > 0 else 0
                accuracy_data[name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                }

            summary = accuracy_data
        elif self.config["evaluation"]["mode"] == "classifier":
            with open(self.run_dir / "eval_logs.jsonl") as f:
                eval_logs = [json.loads(line) for line in f]

            eval_data = {}
            for eval_entry in eval_logs:
                idx = eval_entry["input_idx"]
                name = query_logs[0]["input_name"] if idx == 0 else None
                for log in query_logs:
                    if log["input_idx"] == idx:
                        name = log["input_name"]
                        break
                if name not in eval_data:
                    eval_data[name] = []
                eval_data[name].append(eval_entry["evaluation"])

            for name in inputs_data:
                results = evaluator.aggregate_classifier_results(eval_data[name])
                summary[name] = results
        else:
            for name, logs in inputs_data.items():
                responses = [log["response"] for log in logs]
                results = evaluator.evaluate_frequency(responses)
                summary[name] = results

        model_name = self.config["experiment"].get("model_name", "")
        title_suffix = f" - {model_name}" if model_name else ""

        plot_path = self.run_dir / "plots" / "results.pdf"
        if has_paraphrase:
            from .plotter import plot_scores

            accuracy_values = {k: v["accuracy"] for k, v in summary.items()}
            plot_scores(
                accuracy_values,
                f"Paraphrase Detection Accuracy{title_suffix}",
                plot_path,
                ylabel="Accuracy on Test Examples",
                show_random_baseline=True,
            )
        else:
            if has_demographics:
                plot_stacked_results(
                    summary,
                    f"Results{title_suffix}",
                    plot_path,
                    category_order=["black woman", "white man"],
                    response_order=["No", "Yes"],
                    response_colors={"No": "#FF3F33", "Yes": "#78C841"},
                )
            else:
                plot_stacked_results(summary, f"Results{title_suffix}", plot_path)

        articulation_summaries = {}
        turn_num = 1
        while (self.run_dir / f"articulation_eval_logs_{turn_num}.jsonl").exists():
            with open(self.run_dir / f"articulation_eval_logs_{turn_num}.jsonl") as f:
                articulation_logs = [json.loads(line) for line in f if line.strip()]

            judge_type = self.config["evaluation"]["articulation"]["judge_type"]
            articulation_data = {}
            for log_entry in articulation_logs:
                if has_demographics:
                    key = log_entry.get("demographic", log_entry["input_name"])
                else:
                    key = log_entry["input_name"]
                if key not in articulation_data:
                    articulation_data[key] = []
                articulation_data[key].append(log_entry["answer"])

            articulation_summary = {}
            for key, answers in articulation_data.items():
                valid_answers = [a for a in answers if a is not None]
                if judge_type == "score":
                    avg_score = (
                        sum(valid_answers) / len(valid_answers) if valid_answers else 0
                    )
                    if len(valid_answers) > 1:
                        variance = sum(
                            (x - avg_score) ** 2 for x in valid_answers
                        ) / len(valid_answers)
                        std_score = variance**0.5
                    else:
                        std_score = 0
                    articulation_summary[key] = {
                        "average_score": avg_score,
                        "std_score": std_score,
                        "count": len(valid_answers),
                    }
                elif judge_type == "binary":
                    yes_count = sum(1 for a in valid_answers if a)
                    percent_yes = (
                        (yes_count / len(valid_answers) * 100) if valid_answers else 0
                    )
                    articulation_summary[key] = {
                        "percent_yes": percent_yes,
                        "yes_count": yes_count,
                        "total": len(valid_answers),
                    }

            articulation_plot_path = (
                self.run_dir / "plots" / f"articulation_results_{turn_num}.pdf"
            )
            if judge_type == "score":
                from .plotter import plot_scores

                score_means = {
                    k: v["average_score"] for k, v in articulation_summary.items()
                }
                score_stds = {
                    k: v["std_score"] for k, v in articulation_summary.items()
                }
                if has_demographics:
                    plot_scores(
                        score_means,
                        f"Articulation Results (Turn {turn_num}){title_suffix}",
                        articulation_plot_path,
                        stds=score_stds,
                        category_order=["black woman", "white man"],
                    )
                else:
                    plot_scores(
                        score_means,
                        f"Articulation Results (Turn {turn_num}){title_suffix}",
                        articulation_plot_path,
                        stds=score_stds,
                    )
            else:
                plot_data = {
                    k: {
                        "Yes": v["percent_yes"] / 100,
                        "No": (100 - v["percent_yes"]) / 100,
                    }
                    for k, v in articulation_summary.items()
                }
                if has_demographics:
                    plot_stacked_results(
                        plot_data,
                        f"Articulation Results (Turn {turn_num}){title_suffix}",
                        articulation_plot_path,
                        category_order=["black woman", "white man"],
                        response_order=["No", "Yes"],
                        response_colors={"No": "#d62728", "Yes": "#2ca02c"},
                    )
                else:
                    plot_stacked_results(
                        plot_data,
                        f"Articulation Results (Turn {turn_num}){title_suffix}",
                        articulation_plot_path,
                    )

            articulation_summaries[f"turn_{turn_num}"] = articulation_summary
            turn_num += 1

        full_summary = {"primary": summary, "articulation": articulation_summaries}

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(full_summary, f, indent=2)

    def _create_evaluator(self) -> Evaluator:
        eval_config = self.config["evaluation"]
        mode = eval_config["mode"]

        if mode == "frequency":
            return Evaluator(mode=mode)
        else:
            classifier_template = self._load_file(eval_config["classifier"])
            eval_model = self.config["models"]["eval_model"]
            return Evaluator(
                mode=mode,
                client=self.client,
                eval_model=eval_model,
                classifier_template=classifier_template,
            )
