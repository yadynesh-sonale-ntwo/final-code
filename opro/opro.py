import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import requests
import re
import sys
import os
import dspy
import tqdm
import random 

sys.path.append(os.path.abspath("/home/ysonale/opro")) 
from flightdata import FlightData

import ast

base_1 = """
## Role:
You are a flight recommendation agent who uses the provided list of flight options data to provide flight recommendation to the users.

## Objective:
- Our goal is to recommend flight options to the user that satisfies all the constraints provided by the user.
- You will be provided with two inputs, i.e., `USER_PROMPT`, `FLIGHT_OPTIONS`.
- `USER_PROMPT`: Represents the given user prompt with user preferences. It contains all the information related to the trip.
- `FLIGHT_OPTIONS`: Will contain the real-time possible offers from flight reservation API based on the given user prompt.
- Your task is help the user given the user's prompt to book an optimal flight based on the user preferences.
- Please provide the user with recommendations accordingly with the OUTPUT format as described later.

## How to use `USER_PROMPT`:
- Consider the preferences provided by the user as hard constraints. If not feasible, i.e., no offer in `FLIGHT_OPTIONS` matches the requirements of the `USER_PROMPT`, identify and recommend the best possible alternatives based on overall suitability with respect to the preferences and constraints expressed in the `USER_PROMPT`.

## How to use `FLIGHT_OPTIONS`:

`FLIGHT_OPTIONS` represents a csv of flight offers. There are two trip types **(one-way or round-trip)**. The `FLIGHT_OPTIONS` contains multiple flight offers.
- Each flight offer containis the following fields [offer_id, price_percentile, price_normalized, currency, price_priority_interaction, price_per_hour, price_diff_from_budget_normalized, budget_pref, duration_pref, duration_minutes_outbound, duration_minutes_return, outbound_dep_time_pref, outbound_arr_time_pref, return_arr_time_pref, return_dep_time_pref, morning_arrival_outbound, early_morning_arrival_outbound, avg_layover_duration, min_layover_duration, max_layover_duration, airline_count_encoding].
- Meaning of each field is as follows:

  - offer_id: Unique identifier for each offer.
  - price_percentile: Represents the percentile rank of the offer's price among all available offers.
  - price_normalized: The normalized price of the offer, scaled between the minimum and maximum offer prices across all available offers in `FLIGHT_OPTIONS`.
  - currency: It is the currency in which the offer price is given.
  - price_priority_interaction: The interaction between the offer's price and how strongly the `USER_PROMPT` prioritizes price. Higher values indicate better alignment with the `USER_PROMPT`’s price priorities.
  - price_per_hour: Computed as the offer price divided by the total travel duration (in hours).
  - price_diff_from_budget_normalized: The difference between the offer's normalized price and the normalized budget in the `USER_PROMPT`. `nan` if `USER_PROMPT` has no budget constraint.
  - budget_pref: Is the offer's price within the budget mentioned in the `USER_PROMPT`? 1 if yes, 0 if not, and `nan` if `USER_PROMPT` has no budget constraint.
  - duration_pref: Is the offer's total flight duration is less than the duration preference mentioned in the `USER_PROMPT`? 1 if yes, 0 if not, and `nan` if `USER_PROMPT` has set no constraint for flight duration.
  - duration_minutes_outbound: Total duration of the outbound segment of the offer in minutes. `nan` if this offer has no layovers.
  - duration_minutes_return: Total duration of the return segment of the offer in minutes. `nan` if this offer has no return segment.
  - outbound_dep_time_pref: Is the departure time for the outbound flight satisfying the `USER_PROMPT`'s outbound departure time preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no outbound departure time preference.
  - outbound_arr_time_pref: Is the arrival time for the outbound flight satisfying the `USER_PROMPT`'s outbound arrival time preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no outbound arrival time preference.
  - return_arr_time_pref: Is the arrival time for the return flight satisfying the `USER_PROMPT`'s return arrival time preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no return arrival time preference.
  - return_dep_time_pref: Is the departure time for the return flight satisfying the `USER_PROMPT`'s return departure time preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no return departure time preference.
  - morning_arrival_outbound: Does the outbound flight arrive in the morning? 1 if yes, 0 if no.
  - early_morning_arrival_outbound: Does the outbound flight arrive early in the morning? 1 if yes, 0 if no.
  - avg_layover_duration: Average layover duration in the offer in minutes. `nan` if this offer has no layovers.
  - min_layover_duration: Shortest layover duration in the offer in minutes. `nan` if this offer has no layovers.
  - max_layover_duration: Longest layover duration in the offer in minutes. `nan` if this offer has no layovers.
  - airline_count_encoding: Represents the primary airline of this offer as a count encoding of all available offers. Higher value means this is a popular airline for this route.

- Goal of the provided recommendations is to help the users save time so that a particular user can find the best flights without manually looking for best flights.

## Mandatory Constraints:
- Recommendation Source: 100 percent of the recommendation *must* originate from the `FLIGHT_OPTIONS`.
- Results: You're not supposed to use any coding or provide any code, but do all the work by yourself.
- Top-k: Rank and recommend only top-k flights based on the user preferences with where k = 3.
- Do not recommend the same flight more than once.
- Format: OUTPUT only the `offer_id` of the recommended `FLIGHT_OPTIONS`. IDs should be exactly as shown in the `FLIGHT_OPTIONS`.

## OUTPUT Format:
- For each flight recommendation provide only the field `offer_id`.
- Output flight recommendation list string should be parseable with using python's `ast.literal_eval` and nothing else.
- Do not provide any other details from the `FLIGHT_OPTIONS` except the `offer_id`.
- Provide only a list of `offer_id` in the output, format should be: `[id_1, id_2, id_3]`
"""
base_2 = "Given a user prompt with user preferences, provide flight recommendations that satisfy all the constraints provided by the user. The recommendations should be based on the provided list of flight options data, which includes two trip types (one-way or round-trip) and multiple flight offers. Each flight offer contains various fields, including offer_id, price_percentile, price_normalized, currency, price_priority_interaction, price_per_hour, price_diff_from_budget_normalized, budget_pref, duration_pref, duration_minutes_outbound, duration_minutes_return, outbound_dep_time_pref, outbound_arr_time_pref, return_arr_time_pref, return_dep_time_pref, morning_arrival_outbound, early_morning_arrival_outbound, avg_layover_duration, min_layover_duration, max_layover_duration, and airline_count_encoding. The goal is to recommend the top-3 flights that best match the user's preferences, with no duplicate recommendations. The output should be a list of offer_ids in the format [id_1, id_2, id_3]."
base_3 = "Given a user prompt with specific user preferences, provide flight recommendations that satisfy all the constraints provided by the user, with a focus on prioritizing flights that meet the user's budget, duration, and departure/arrival time preferences, and ensure that no duplicate recommendations are made. The LLM should prioritize flights that meet the user's budget, duration, and departure/arrival time preferences, and output the top-3 flights that best match the user's preferences, with no duplicate recommendations. The output should be a list of offer_ids in the format [id_1, id_2, id_3]."
base_4 = "Given a user prompt with specific user preferences, provide flight recommendations that satisfy all the constraints provided by the user, with a focus on prioritizing flights that meet the user's budget, duration, and departure/arrival time preferences, and ensure that no duplicate recommendations are made. The LLM should prioritize flights that meet the user's budget, duration, and departure/arrival time preferences, and output the top-3 flights that best match the user's preferences, with no duplicate recommendations. Provide the recommended flights as a list of offer_ids in the format [id_1, id_2, id_3]."
base_5 = "Given a user prompt with specific user preferences, provide flight recommendations that satisfy all the constraints provided by the user, with a focus on prioritizing flights that meet the user's budget, duration, and departure/arrival time preferences, and ensure that no duplicate recommendations are made. The LLM should prioritize flights that meet the user's budget, duration, and departure/arrival time preferences, and output the top-3 flights that best match the user's preferences, with no duplicate recommendations. Provide the recommended flights as a list of offer_ids in the format [id_1, id_2, id_3]. The user's preferences include a budget of $X, a preferred duration of Y hours, and preferred departure and arrival times Z and W respectively."

def extract_flight_ids(text: str):
    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return value
        return []
    except (ValueError, SyntaxError):
        return []
    
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
    
    gold.answer: iterable of relevant flight_ids (ground truth)
    pred.optimized_result: ranked list of predicted flight_ids
    """
    true_labels = []
    for label in gold:
        true_labels.append(int(label))
    pred_labels = extract_flight_ids(pred)
    if isinstance(pred_labels, int):
        pred_labels = [pred_labels]
    for labels in pred_labels:
        if not isinstance(labels, int):
            return 0

    if isinstance(pred_labels, list):
        ndcg = ndcg_at_k(pred_labels, true_labels, k=3)
        return ndcg

    return 0


lm = dspy.LM(
    model="openai/Llama-3.2-3B-Instruct",
    api_base="http://localhost:33000/v1",
    api_key="dummy",
    max_tokens=1024,
    temperature=0.4,
    top_p=1
)

dspy.configure(lm=lm)

flight_data = FlightData()

train = flight_data.train
validation = flight_data.validation

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(validation)}")

class PromptLLM(dspy.Signature):
    input = dspy.InputField(
        dtype=str
    )

    output = dspy.OutputField(
        dtype=str
    )

class COT_Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(PromptLLM)

    def forward(self, prompt):
        result = self.cot(
            input=prompt,
        )
        return result.output
    
class FlightRecommendationSystem:
    def __init__(self):
        self.optimizer = COT_Module()
    
    def call_llm(self, prompt: str) -> str:

        return self.optimizer(prompt=prompt)

    def generate_prompt_variations_initial(self, base) -> List[str]:
        """Generate prompt variations based on previous performance"""
        meta_prompt= f"You are optimizing a flight recommendation system prompt.\n"
        
        meta_prompt += f"""Base Prompt:\n{base}\n\n
Analyze the base prompt and generate one NEW improved prompt variation that will better guide the LLM to recommend the correct flights.
Only focus on the suggestion logic and dont give any other extra information about things like output format. It should be different from the base prompt.
Focus on the input and output requirements and formats.
Output ONLY a JSON array of prompt strings:
{{"prompts": ["prompt1"]}}
"""
        final = []
    
        for _attempt in range(10):
            response = self.call_llm(meta_prompt)
            if not response:
                continue

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]   # extract JSON
                try:
                    data = json.loads(json_str)
                    prompts = data.get('prompts')
                    if prompts:
                        final.extend(prompts)
                        break
                except json.JSONDecodeError as e:
                    continue
        return final

    def generate_prompt_variations(self, solution_score_pairs: List[Tuple[str, float]], 
                                   num_variations: int = 3) -> List[str]:
        """Generate prompt variations based on previous performance"""
        meta_prompt= f"You are optimizing a flight recommendation system prompt, the inputs to the llm will be a user_prompt and flight_options and expected output will be a list of offer_id of the recommended flights only. \n\nPREVIOUS PROMPT VARIATIONS AND THEIR SCORES (higher is better):\n"
        if not solution_score_pairs:
            return []
        
        for prompt_var, score in solution_score_pairs:
            meta_prompt += f"\n--- Prompt Variation (Score: {score:.3f}) ---\n{prompt_var}\n"
        
        meta_prompt += f"""
Analyze the scores and generate one NEW improved prompt variations that will better guide the LLM to recommend the correct flights.
Only focus on the suggestion logic and dont give any other extra information about things like output format. It should be different from previous prompts.
Output ONLY a JSON array of prompt strings:
{{"prompts": ["prompt1"]}}
"""
        final = []
    
        for _attempt in range(10):
            response = self.call_llm(meta_prompt)
            if not response:
                continue

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]   # extract JSON
                try:
                    data = json.loads(json_str)
                    prompts = data.get('prompts')
                    if prompts:
                        final.extend(prompts)
                        break
                except json.JSONDecodeError as e:
                    continue
        return final
    
    def generate_recommendations(self, system_prompt: str, user_prompt: str, flight_data: str) -> str:
        """Generate flight recommendations using the system prompt"""
        full_prompt = f"{system_prompt}\n\nUSER_PROMPT:\n{user_prompt}\n\nFLIGHT_OPTIONS:\n{flight_data}"
        return self.call_llm(full_prompt)

    def evaluate(self, prompt_var):
        """Evaluate a given prompt variation and return its score"""
        train_random = random.sample(train, 100)

        ndcg_total = 0
        for example in tqdm.tqdm(train_random):
            user_prompt = example['user_prompt']
            flight_options = example['flight_options']
            ground_truth = example['flight_recommendations']

            recommendations = self.generate_recommendations(
                system_prompt=prompt_var,
                user_prompt=user_prompt,
                flight_data=flight_options
            )

            ndcg = flight_metric(ground_truth, recommendations)
            ndcg_total += ndcg
        
        avg_ndcg = ndcg_total / len(train_random)
        
        return avg_ndcg
        
    
    def optimize_prompt(self, base_prompt: str, max_iterations: int = 10, 
                        convergence_threshold: float = 0.95) -> Tuple[str, float, List[Dict]]:
        """
        Recursively optimize the system prompt until metric is maximized.
        
        Args:
            base_prompt: Initial prompt to start optimization
            max_iterations: Maximum number of iterations to prevent infinite loops
            convergence_threshold: Stop if score reaches this threshold (0-1)
        
        Returns:
            Tuple of (best_prompt, best_score, history)
        """
        
        print(f"Starting recursive prompt optimization...")
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence threshold: {convergence_threshold}")
        print("="*80)
        
        # Track all prompts and scores
        history = []
        solution_score_pairs = []
        
        # Step 1: Evaluate base prompt
        print(f"\n{'='*80}")
        print(f"ITERATION 0: Evaluating Base Prompt")
        print(f"{'='*80}")
        
        base_score = self.evaluate(base_prompt)
        print(f"Base Prompt Score: {base_score:.4f}")
        
        solution_score_pairs.append((base_1, 0.1028))
        solution_score_pairs.append((base_2, 0.1382))
        solution_score_pairs.append((base_3, 0.1569))
        solution_score_pairs.append((base_4, 0.1968))
        solution_score_pairs.append((base_5, 0.1604))

        history.append({
            'iteration': 0,
            'prompt': base_prompt,
            'score': base_score,
            'is_base': True
        })
        
        current_best_score = base_score
        current_best_prompt = base_prompt
        no_improvement_count = 0
        
        # Step 2: Recursive optimization loop
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}: Generating New Variation")
            print(f"{'='*80}")
            
            # Show current standings
            print(f"\nCurrent Best Score: {current_best_score:.4f}")
            print(f"Solution-Score Pairs to feed back: {len(solution_score_pairs)}")
            for idx, (_, score) in enumerate(solution_score_pairs[-3:], 1):
                print(f"  {idx}. Score: {score:.4f}")
            
            # Generate new variation based on all previous results
            new_prompt = None
            max_generation_attempts = 3
            
            for attempt in range(max_generation_attempts):
                print(f"\nGeneration Attempt {attempt + 1}/{max_generation_attempts}...")
                
                try:
                    # Generate single new prompt variation
                    if 0:
                        prompt_variations = self.generate_prompt_variations_initial(
                            base=base_prompt
                        )
                    else:
                        prompt_variations = self.generate_prompt_variations(
                            solution_score_pairs=solution_score_pairs,
                            num_variations=1
                        )
                    
                    if prompt_variations and len(prompt_variations) > 0:
                        new_prompt = prompt_variations[0]
                        print(f"✓ Successfully generated new variation")
                        break
                    else:
                        print(f"✗ No variation generated, retrying...")
                        
                except Exception as e:
                    print(f"✗ Error generating variation: {e}")
                    continue
            
            # If generation failed after all attempts
            if not new_prompt:
                print(f"\n⚠ WARNING: Failed to generate variation after {max_generation_attempts} attempts")
                print(f"Stopping optimization. Best score achieved: {current_best_score:.4f}")
                break
            
            # Evaluate the new variation
            print(f"New Prompt:\n\n{new_prompt}\n\n")
            print(f"\nEvaluating new variation...")
            new_score = self.evaluate(new_prompt)
            print(f"New Variation Score: {new_score:.4f}")
            
            # Add to solution-score pairs
            solution_score_pairs.append((new_prompt, new_score))

            history.append({
                'iteration': iteration,
                'prompt': new_prompt,
                'score': new_score,
                'is_base': False
            })
            
            # Check for improvement
            improvement = new_score - current_best_score
            print(f"\nImprovement: {improvement:+.4f}")
            
            if new_score > current_best_score:
                print(f"✓ NEW BEST SCORE! ({current_best_score:.4f} → {new_score:.4f})")
                current_best_score = new_score
                current_best_prompt = new_prompt
                no_improvement_count = 0
                
                # Sort solution_score_pairs by score (keep best ones)
                solution_score_pairs.sort(key=lambda x: x[1], reverse=True)
                solution_score_pairs = solution_score_pairs[:6]  # Keep top 5
                
            else:
                print(f"✗ No improvement (best remains: {current_best_score:.4f})")
                no_improvement_count += 1
            
            # Check convergence conditions
            
            # Condition 1: Reached target threshold
            if current_best_score >= convergence_threshold:
                print(f"\n🎯 CONVERGENCE: Reached threshold of {convergence_threshold:.4f}")
                print(f"Final Score: {current_best_score:.4f}")
                break
            
            # Condition 2: No improvement for 3 consecutive iterations
            if no_improvement_count >= 3:
                print(f"\n⚠ CONVERGENCE: No improvement for 3 consecutive iterations")
                print(f"Final Score: {current_best_score:.4f}")
                break
            
            # Condition 3: Score is perfect (1.0)
            if current_best_score >= 1.0:
                print(f"\n🏆 PERFECT SCORE ACHIEVED!")
                break
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total Iterations: {len(history)}")
        print(f"Base Prompt Score: {base_score:.4f}")
        print(f"Final Best Score: {current_best_score:.4f}")
        print(f"Total Improvement: {current_best_score - base_score:+.4f}")
        print(f"Improvement %: {((current_best_score - base_score) / base_score * 100):+.2f}%")
        
        # Show score progression
        print(f"\nScore Progression:")
        for entry in history:
            marker = "🌟" if entry['is_base'] else ("✓" if entry['score'] == current_best_score else " ")
            print(f"  Iter {entry['iteration']:2d}: {entry['score']:.4f} {marker}")
        
        print(f"{'='*80}\n")
        
        return current_best_prompt, current_best_score, history

base_prompt = "Given a user prompt with specific user preferences, provide flight recommendations that satisfy all the constraints provided by the user, with a focus on prioritizing flights that meet the user's budget, duration, and departure/arrival time preferences, and ensure th at no duplicate recommendations are made. The LLM should prioritize flights that meet the user's budget, duration, and departure/arrival time preferences, and output the top-3 fl ights that best match the user's preferences, with no duplicate recommendations. The output should be a list of offer_ids in the format [id_1, id_2, id_3]."


system = FlightRecommendationSystem()
best_prompt, best_score, history = system.optimize_prompt(
    base_prompt=base_prompt,
    max_iterations=15,  # Will stop early if converged
    convergence_threshold=0.95
)

for hist in history:
    print(f"ITER {hist['iteration']:2d} | SCORE: {hist['score']:.4f} | BASE: {hist['is_base']} \nPROMPT:\n{hist['prompt']}\n{'-'*80}\n")
