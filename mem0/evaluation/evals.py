import argparse
import concurrent.futures
import json
import threading
from collections import defaultdict

from dotenv import load_dotenv
from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm

# 加载 .env 文件中的环境变量
load_dotenv()


def process_item(item_data):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        # Skip category 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

    return local_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument(
        "--input_file", type=str, default="results/rag_results_500_k1.json", help="Path to the input dataset file"
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation_metrics.json", help="Path to save the evaluation results"
    )
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, item_data) for item_data in data.items()]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    # Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")
    
    # 计算并打印三种平均 score
    all_bleu_scores = []
    all_f1_scores = []
    all_llm_scores = []
    
    for k, items in results.items():
        for item in items:
            all_bleu_scores.append(item["bleu_score"])
            all_f1_scores.append(item["f1_score"])
            all_llm_scores.append(item["llm_score"])
    
    if all_bleu_scores:
        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
        avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
        avg_llm = sum(all_llm_scores) / len(all_llm_scores)
        
        print("\n" + "=" * 50)
        print("Average Scores:")
        print(f"  BLEU Score: {avg_bleu:.4f}")
        print(f"  F1 Score:   {avg_f1:.4f}")
        print(f"  LLM Score:  {avg_llm:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
