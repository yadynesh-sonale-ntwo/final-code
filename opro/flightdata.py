import random
import sys
import os
import tqdm
import numpy as np
from sklearn.metrics import ndcg_score

sys.path.append(os.path.abspath("/home/ysonale/opro")) 
from load_annotations import load_val, load_train, extract_flight_ids


class FlightData:
    def __init__(self):
        self.do_shuffle = False

        dataset_train = load_train()
        dataset_validation = load_val()      

        official_train = []
        official_validation = []

        for example in tqdm.tqdm(dataset_train):
            user_prompt = example["nlq"]
            flight_options = example["data"]
            flight_recommendations = example["labels"]

            official_train.append({"user_prompt": user_prompt, "flight_options": flight_options, "flight_recommendations": flight_recommendations})

        for example in tqdm.tqdm(dataset_validation):
            user_prompt = example["nlq"]
            flight_options = example["data"]
            flight_recommendations = example["labels"]

            official_validation.append({"user_prompt": user_prompt, "flight_options": flight_options, "flight_recommendations": flight_recommendations})

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_validation)

        trainset = official_train[:]
        valset = official_validation[:]

        import dspy

        trainset = [
            dspy.Example(**x).with_inputs("user_prompt", "flight_options")
            for x in trainset
        ]

        valset = [
            dspy.Example(**x).with_inputs("user_prompt", "flight_options")
            for x in valset
        ]

        self.train = trainset
        self.validation = valset



def dcg_at_k(pred_ids, true_ids, k):
    true_set = set(true_ids)
    dcg = 0.0
    for i, pred_id in enumerate(pred_ids[:k], start=1):
        if pred_id in true_set:
            relevance = 1  # binary relevance
        else:
            relevance = 0
        dcg += relevance / np.log2(i + 1)
    return dcg

def ndcg_at_k(pred_ids, true_ids, k):
    dcg = dcg_at_k(pred_ids, true_ids, k)
    # ideal: top k relevant items first
    ideal_order = (true_ids[:k] + pred_ids)[:k]  # fallback, but simpler:
    # Actually, IDCG should be computed by putting all relevant items first.
    # For binary relevance, IDCG = sum_{i=1}^{min(k, num_relevant)} 1 / log2(i+1)
    idcg = dcg_at_k(true_ids, true_ids, k)  # This works IF relevance scoring in dcg_at_k uses binary
    # but if using binary, dcg_at_k(true_ids, true_ids, k) assumes true_ids list has all relevance=1
    # So better: idcg = sum(1 / log2(i+1) for i in range(1, min(k, len(true_ids)) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def flight_metric(gold, pred, trace=None):
    """
    Computes NDCG for flight recommendation.
    
    gold.flight_recommendations: iterable of relevant flight_ids (ground truth)
    pred.optimized_result: ranked list of predicted flight_ids
    """
    true_labels = []
    for label in gold.flight_recommendations:
        true_labels.append(int(label))
    pred_labels = extract_flight_ids(pred.optimized_result)
    for labels in pred_labels:
        if not isinstance(labels, int):
            return 0

    if isinstance(pred_labels, list):
        ndcg = ndcg_at_k(pred_labels, true_labels, k=3)
        return ndcg

    return 0