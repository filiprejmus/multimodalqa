from evaluate import evaluate_predictions
import json


def evaluate_results():
    results = json.load(open("inference_results.json"))
    gold_answers = {}
    predicted_answers = {}
    cost = []
    for r in results:
        qid = r["qid"]
        expected_answers = r["expected_answers"]
        infered_answers = r["infered_answers"]
        # Assuming expected_answers is a list and we want the first answer
        gold_answers[qid] = [str(answer) for answer in expected_answers]
        predicted_answers[qid] = [str(answer) for answer in infered_answers]
        cost.append(r["price_in_usd"])

    print("Cost: ", str(sum(cost)) + " USD")
    print("average cost: ", str((sum(cost) / len(cost)) * 100) + " Cents")
    return evaluate_predictions(predicted_answers, gold_answers)


eval_scores, instance_eval_results = evaluate_results()
print(eval_scores)
# print(instance_eval_results)
