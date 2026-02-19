import os
from dotenv import load_dotenv
import numpy as np
import wandb

load_dotenv()
wandb.login(key=os.getenv("API_WANDB"))

class ThresholdTuner:
    def __init__(self, ground_truth, outputs, evaluate_fn, prec_w=0.5, reca_w=0.5):
        self.ground_truth = ground_truth
        self.outputs = outputs
        self.evaluate = evaluate_fn
        self.thresholds = np.linspace(0.2, 1.0, 30)
        self.prec_w = prec_w
        self.reca_w = reca_w 
        self.positives = float(len(ground_truth))

    def tune(self):

        wandb.init(entity="grumpy_ananas-none", project="RAG", reinit='finish_previous')

        best_score = -1
        best_threshold = None

        for threshold in self.thresholds:
            retrieved = 0

            tp = fp = fn = 0

            for outs in self.outputs:
                current_outs = outs.copy()

                if outs["score"] < threshold:
                    current_outs["document_chunk"] = ""
                    current_outs["answer"] = "I don't know. No relevant documents were retrieved."
                    continue

                retrieved += 1

            tp =  self.positives
            fp = retrieved - self.positives if retrieved > self.positives else 0.0
            fn = tp - retrieved if retrieved < self.positives else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            weighted_score = self.prec_w * precision + self.reca_w * recall

            wandb.log({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "weighted_score": weighted_score
            })

            if f1 > best_score:
                best_score = f1
                best_threshold = threshold

        wandb.log({
            "best_threshold": best_threshold,
            "best_f1": best_score,
            "weighted_score": weighted_score
        })

        wandb.finish()

        return best_threshold, best_score
