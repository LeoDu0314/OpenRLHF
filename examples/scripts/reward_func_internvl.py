import os
import re
from datetime import datetime

import regex
import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")


def accuracy_reward_func(completion, answer):
    answers = [
        answer,
        answer.removeprefix("\\boxed{").removesuffix("}"),
        answer.removeprefix("$$").removeprefix("$").removesuffix("$$").removesuffix("$"),
    ]
    completion_match = regex.findall(
        r"(\\boxed\{(?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*\})", completion, re.DOTALL
    )
    completion = completion_match[-1][0].strip() if completion_match else completion.strip()
    completion_raw = completion.removeprefix("\\boxed{").removesuffix("}")

    reward = 0.0

    if completion in answers:
        reward = 1.0
    elif completion_raw in answers:
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
            completion_raw = parse(
                completion_raw,
                extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
            )
            if verify(answer, completion):
                reward = 1.0
            elif verify(answer, completion_raw):
                reward = 1.0
        except:
            pass

    return reward, completion_match


def format_reward_func(completion):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"<think>.+?</think>.+$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 1.0 if matches else 0.0


def reward_func(queries, prompts):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    pattern = r"<\|im_start\|>\s*assistant(.*?)<\|im_end\|>"

    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt in zip(queries, prompts):
            try:
                query = re.sub(r"\s*<IMG_CONTEXT>\s*", "", query)
                query = re.sub(r"<img>\s*</img>", " <image>", query)
                query = re.sub("</s>", "", query)
                response = re.search(pattern, query, re.DOTALL).group(1).strip()
            except Exception as e:
                print("Error in reward_func: ", e)
                response = ""

            answer = prompt["answer"]

            try:
                accuracy_reward, completion_match = accuracy_reward_func(response, answer)
                format_reward = format_reward_func(response)
                reward = accuracy_reward + format_reward * 0.5
            except Exception as e:
                print("Error in reward_func: ", e)
                accuracy_reward = 0.0
                format_reward = 0.0
                reward = 0.0
                completion_match = []

            rewards.append(reward)
            accuracy_rewards.append(accuracy_reward)
            format_rewards.append(format_reward)
            f.write(f"===============================================================\n")
            f.write("Query: " + query + "\n")
            f.write("Completion Match: " + str(completion_match) + "\n")
            f.write("Answer: " + answer + "\n")
            f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
            f.write(f"===============================================================\n")

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }
