#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dspy
import pandas as pd 
import sys
import os
import json
from dspy.teleprompt import MIPROv2
import sys

sys.path.append(os.path.abspath("/home/ysonale/PromptMatrix/mipro_features")) 
from flightdata import FlightData, flight_metric


flight_data = FlightData()


print(len(flight_data.train))
print(len(flight_data.validation))


print(flight_data.train[140]['flight_options'])


print(flight_data.train[0]['user_prompt'])


print(len(flight_data.train[140]['flight_options'][0].keys()))

class FlightOptimizerSignature(dspy.Signature):
    r"""
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
- Each flight offer containis the following fields [offer_id, price_percentile, price_normalized, currency, price_priority_interaction, price_per_hour, price_diff_from_budget_normalized, budget_pref, duration_pref, outbound_dep_time_pref, outbound_arr_time_pref, morning_arrival_outbound, early_morning_arrival_outbound, avg_layover_duration, min_layover_duration, max_layover_duration, airline_count_encoding].
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
  - outbound_dep_time_pref: Is the departure time for the outbound flight satisfying the `USER_PROMPT`'s outbound departure time preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no outbound departure time preference.
  - outbound_arr_time_pref: Is the arrival time for the outbound flight satisfying the `USER_PROMPT`'s outbound arrival time preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no outbound arrival time preference.
  - morning_arrival_outbound: Does the outbound flight arrive in the morning? 1 if yes, 0 if no.
  - early_morning_arrival_outbound: Does the outbound flight arrive early in the morning? 1 if yes, 0 if no.
  - avg_layover_duration: Average layover duration in the offer in minutes. `nan` if this offer has no layovers.
  - min_layover_duration: Shortest layover duration in the offer in minutes. `nan` if this offer has no layovers.
  - max_layover_duration: Longest layover duration in the offer in minutes. `nan` if this offer has no layovers.
  - airline_pref: Is the offer's airline satisfying the `USER_PROMPT`'s airline preference? 1 if yes, 0 if not, and `nan` if the `USER_PROMPT` has no airline preference.
  - airline_count_encoding: Represents the primary airline of this offer as a count encoding of all available offers. Higher value means this is a popular airline for this route.

- Goal of the provided recommendations is to help the users save time so that a particular user can find the best flights without manually looking for best flights.

## Mandatory Constraints:
- Recommendation Source: 100 percent of the recommendation *must* originate from the `FLIGHT_OPTIONS`.
- Results: You're not supposed to use any coding or provide any code, but do all the work by yourself.
- Top-k: Rank and recommend only top-k flights based on the user preferences with where k = 3.
- Do not recommend the same flight more than once.
- Format: OUTPUT only the `offer_id` of the recommended `FLIGHT_OPTIONS`. IDs should be exactly as shown in the `FLIGHT_OPTIONS`.

## OUTPUT Format:
- For each flight recommendation provide only the field offer_id.
- Output flight recommendation list string should be parseable with using python's `ast.literal_eval` and nothing else.
- Do not provide any other details from the `FLIGHT_OPTIONS` except the `offer_id`.
- Provide only a list of `offer_id` in the output, format should be: `[id_1, id_2, id_3]`
    """

    user_prompt = dspy.InputField(
        dtype=str,
        desc="The user's prompt describing their flight preferences."
    )

    flight_options = dspy.InputField(
        dtype=list,
        desc="A list of available flight options to choose from."
    )

    optimized_result = dspy.OutputField(
        dtype=list,
        desc="The output is the flight option(s) selected by the optimizer."
    )


# In[10]:


lm = dspy.LM(
    model="openai/Llama-3.2-3B-Instruct",
    api_base="http://localhost:30000/v1",
    api_key="dummy",
    max_tokens=1024,
    temperature=0.7,
    top_p=1
)

dspy.configure(lm=lm)

# Initialize optimizer
teleprompter = MIPROv2(
    metric=flight_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
    verbose=0
)

# Optimize program
print(f"Optimizing program with MIPROv2...")
optimized_program = teleprompter.compile(
    dspy.ChainOfThought(FlightOptimizerSignature),
    trainset=flight_data.train,  # Replace with csv, toon, or json
    valset=flight_data.validation,     # Replace with csv, toon, or json
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
)

# Save optimize program for future use
optimized_program.save(f"optimized.json")




