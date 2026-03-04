import os
from dotenv import load_dotenv
import numpy as np
import wandb

load_dotenv()
wandb.login(key=os.getenv("API_WANDB"))

class ThresholdTuner:
    def __init__(self, ground_truth, outputs, prec_w=0.5, reca_w=0.5):
        self.ground_truth = ground_truth
        self.outputs = outputs
        self.thresholds = np.linspace(0.2, 1.0, 100)
        self.prec_w = prec_w
        self.reca_w = reca_w 
        self.positives = float(len(ground_truth))

    def tune(self):

        wandb.init(entity=os.getenv("WANDB_NAME"), project=os.getenv("WANDB_PROJECT"), reinit='finish_previous')

        best_score = -1
        best_threshold = None

        for threshold in self.thresholds:
            retrieved = 0

            tp = fp = fn = 0

            for entry in self.outputs:
                prediction = entry[0]
                ground_truth_list = entry[1]

                if prediction["answer"] == "I don't know. No relevant documents were retrieved.":
                    continue
                elif prediction["score"] < threshold:
                    prediction["document_chunk"] = ""
                    prediction["answer"] = "I don't know. No relevant documents were retrieved."
                    continue
                for res_gt in ground_truth_list:
                    if res_gt["correct"] == True:
                        retrieved += 1
                    else:
                        continue 

            tp = self.positives
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
