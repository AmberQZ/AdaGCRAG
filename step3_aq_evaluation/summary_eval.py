# This code is based on the Summary Evaluation in LightRAG codebase.

import re
import os
import time
import json
import jsonlines
import argparse
import pandas as pd
from openai import OpenAI
import multiprocessing as mp
from functools import partial


INCLUDE_COL = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Directness",
    "Overall Winner",
]



def eval_single(i, query, answer1, answer2, args):
    client = OpenAI(api_key="sk-", base_url="")
    if i % 10 == 0:
        print(f"Processing {i}.")

    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """

    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**,**Empowerment**, and **Directness**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
    - **Directness**. How specifically and clearly does the answer address the question?
    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these four categories.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the four criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence explanation here]"
        }},
        "Diversity": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence  explanation here]"
        }},
        "Empowerment": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence  explanation here]"
        }},
        "Directness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide one sentence  explanation here]"
        }},
        "Overall Winner": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Briefly summarize why this answer is the overall winner based on the three criteria]"
        }}
    }}
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    # 准备用于传递给 API 的参数字典
    parameters = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4000,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
    }
    max_retries = 3
    retries = 0
    success = False
    result = None
    res_dict = {}
    token_usage = 0
    while not success and retries < max_retries:
        try:
            response = client.chat.completions.create(**parameters)
            result = response.choices[0].message.content
            token_usage += response.usage.total_tokens
        except Exception as e:
            retries += 1
            print(f"OpenAI error, retrying... ({retries}/{max_retries})")
            time.sleep(2)

        try:
            result = re.sub(r"\n+", "\n", result)

            json_res = json.loads(result)

            col_check = True
            for col in INCLUDE_COL:
                if col not in json_res:
                    print(
                        f"Error parsing JSON response from OpenAI. not include col: {col}"
                    )
                    col_check = False
                    break
                if "Winner" not in json_res[col]:
                    print(
                        f"Error parsing JSON response from OpenAI. not include winner in col: {col}"
                    )
                    col_check = False
                    break
            if not col_check:
                retries += 1
                continue
            res_dict = {
                "Comprehensiveness": json_res["Comprehensiveness"]["Winner"],
                "Diversity": json_res["Diversity"]["Winner"],
                "Empowerment": json_res["Empowerment"]["Winner"],
                "Directness": json_res["Directness"]["Winner"],
                "Overall Winner": json_res["Overall Winner"]["Winner"],
                "ori_json_res": json_res,
            }

        except Exception as e:
            print("Error parsing JSON response from OpenAI.")
            print(e)
            retries += 1
            continue

        success = True

    if not success:
        print("Failed to get response from OpenAI.")
        return (
            {
                "Comprehensiveness": "N/A",
                "Diversity": "N/A",
                "Empowerment": "N/A",
                "Directness": "N/A",
                "Overall Winner": "N/A",
                "ori_json_res": "N/A",
            },
        )
    return i, res_dict, token_usage


def batch_eval(df_1, df_2, args):
    queries = df_1["question"].tolist()
    answers1 = df_1["output"].tolist()
    answers2 = df_2["output"].tolist()

    eval_tuples = list(zip(queries, answers1, answers2))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        process_func = partial(eval_single, args=args)
        results = pool.starmap(
            process_func, [(i, *eval_tuple) for i, eval_tuple in enumerate(eval_tuples)]
        )

    results_all_list = []
    for num_i, result, token_usage in results:  # 解包第三个值 token_usage
        query = queries[num_i]
        answer1 = answers1[num_i]
        answer2 = answers2[num_i]
        result["query"] = query
        result["answer1"] = answer1
        result["answer2"] = answer2
        result["id"] = num_i
        result["token_usage"] = token_usage  # 可选：记录 token 使用量
        results_all_list.append(result)

    res_df = pd.DataFrame(results_all_list)
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file1",
        type=str,
        help="Path to the first input JSON file",
    )

    parser.add_argument(
        "--input_file2",
        type=str,
        help="Path to the second input JSON file",
    )

    # Ensure this is defined only once
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Path to the output CSV file",
    )

    args = parser.parse_args()

    # 加载 JSON 文件
    with open(args.input_file1, "r", encoding="utf-8") as f1, open(args.input_file2, "r", encoding="utf-8") as f2:
        data1 = [json.loads(line) for line in f1]
        data2 = [json.loads(line) for line in f2]

    eval_file1 = pd.DataFrame(data1)
    eval_file2 = pd.DataFrame(data2)

    print(f"shape1: {eval_file1.shape}, shape2: {eval_file2.shape}")

    # 批量评估
    res_df = batch_eval(eval_file1, eval_file2, args)

    # 保存结果
    res_df.to_csv(args.output_file_name, index=False)
    print(f"Results saved to {args.output_file_name}.")

    for col in INCLUDE_COL:
        win1_times = res_df[col].value_counts().get("Answer 1", 0)
        print(
            f"{col}: Answer 1 wins {win1_times} / 125 times, { 100 * (win1_times / 125) :.2f}",
            end=" ",
        )
        print(
            f"Answer 2 wins {125-win1_times} / 125 times, {100 * ((125-win1_times)/125):.2f}"
        )    