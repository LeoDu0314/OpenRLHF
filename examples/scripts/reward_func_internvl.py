import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")


def accuracy_reward_func(completion, answer):
    completion_match = re.search(r"\\boxed\{(.*?)\}\s*$", completion, re.DOTALL)
    completion = completion_match.group(0).strip() if completion_match else completion.strip()

    reward = 0.0

    if completion == answer:
        reward = 1.0
    else:
        try:
            answer = parse(
                answer, extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()]
            )
            completion = parse(
                completion,
                extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
            )
            if verify(answer, completion):
                reward = 1.0
        except:
            pass

    return reward


def format_reward_func(completion):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"<think>.*?</think>"
        r".*\\boxed\{.*\}\s*$"
    )
    # pattern = r"^<think>.*?</think>(?:(?!<\/?think>).)*\\boxed\{.*\}\s*$"
    matches = re.search(pattern, completion, re.DOTALL)
    return 1.0 if matches else 0.0


def reward_func(queries, prompts):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    pattern = r"<\|im_start\|> assistant(.*?)<\|im_end\|>"

    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt in zip(queries, prompts):
            query = re.sub(r"\s*<IMG_CONTEXT>\s*", "", query)
            query = re.sub(r"<img>\s*</img>", " <image>", query)
            query = re.sub("</s>", "", query)
            try:
                response = re.search(pattern, query, re.DOTALL).group(1).strip()
                answer = prompt["answer"]

                accuracy_reward = accuracy_reward_func(response, answer)
                format_reward = format_reward_func(response)
            except Exception as e:
                print(e)
                accuracy_reward = 0.0
                format_reward = 0.0

            rewards.append(accuracy_reward + 0.5 * format_reward)
            f.write(f"===============================================================\n")
            f.write("Query: " + query + "\n")
            f.write("Answer: " + answer + "\n")
            f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
            f.write(f"===============================================================\n")

    return torch.tensor(rewards, dtype=torch.float32)
