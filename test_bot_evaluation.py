import time
from brain import preprocess
from embeddings import initialise, classify_intent, debug_similarities
from test_dataset import TEST_DATA

def run_evaluation():
    print("=" * 60)
    print("Initializing Model for Evaluation...")
    # This loads the 1.1GB model into memory just like normal startup
    initialise()
    print("=" * 60)
    
    total_examples = 0
    correct_predictions = 0
    failures = []
    successes = []
    
    intent_stats = {}
    
    print("\nStarting Evaluation against Holdout Dataset...\n")
    
    start_time = time.time()
    
    for expected_intent, phrases in TEST_DATA.items():
        intent_stats[expected_intent] = {"total": 0, "correct": 0}
        
        for phrase in phrases:
            total_examples += 1
            intent_stats[expected_intent]["total"] += 1
            
            # Step A: Preprocess exactly as the bot does in real life
            cleaned_text = preprocess(phrase)
            
            # Step B: Classify intent
            predicted_intent = classify_intent(cleaned_text)
            
            # Step C: Evaluate Prediction
            scores = debug_similarities(cleaned_text)
            result_data = {
                "phrase": phrase,
                "expected": expected_intent,
                "predicted": predicted_intent,
                "scores": scores
            }
            
            if predicted_intent == expected_intent:
                correct_predictions += 1
                intent_stats[expected_intent]["correct"] += 1
                successes.append(result_data)
            else:
                failures.append(result_data)
                
    elapsed_time = time.time() - start_time
    
    # --- REPORTING ---
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Examples Tested : {total_examples}")
    print(f"Total Correct         : {correct_predictions}")
    print(f"Overall Accuracy      : {(correct_predictions / total_examples) * 100:.2f}%\n")
    print(f"Average Latency       : {(elapsed_time / total_examples) * 1000:.1f} ms per message")
    
    print("\n--- Accuracy by Intent ---")
    for intent, stats in intent_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{intent:<15}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
        
    print("\n" + "=" * 60)
    print("SUCCESS ANALYSIS (Correct Classifications)")
    print("=" * 60)
    
    if not successes:
        print("No successful classifications.")
    else:
        for success in successes:
            print(f"\n[✓] Phrase: '{success['phrase']}'")
            print(f"    Expected : {success['expected']}")
            print(f"    Predicted: {success['predicted']}")
            
            sorted_scores = sorted(success['scores'].items(), key=lambda x: x[1], reverse=True)
            print("    Top 3 Model Scores:")
            for intent, score in sorted_scores[:3]:
                threshold_flag = " [OVER THRESHOLD]" if score >= 0.42 else ""
                print(f"      - {intent:<15}: {score:.4f}{threshold_flag}")

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS (Misclassifications)")
    print("=" * 60)
    
    if not failures:
        print("Perfect classification! No failures detected.")
    else:
        for fail in failures:
            print(f"\n[X] Phrase: '{fail['phrase']}'")
            print(f"    Expected : {fail['expected']}")
            print(f"    Predicted: {fail['predicted']}")
            
            sorted_scores = sorted(fail['scores'].items(), key=lambda x: x[1], reverse=True)
            print("    Top 3 Model Scores:")
            for intent, score in sorted_scores[:3]:
                threshold_flag = " [OVER THRESHOLD]" if score >= 0.42 else ""
                print(f"      - {intent:<15}: {score:.4f}{threshold_flag}")

if __name__ == "__main__":
    run_evaluation()
