import json
import os
import re
import pandas as pd
import ast
import tqdm

def str_list_to_int_list(lst):
    return [int(x) for x in lst]

def extract_flight_ids(text: str):
    out = []
    try:
        out = ast.literal_eval(text)
    except:
        out = []
    return out

def get_label(annotation, type_):
    path = f"/home/common/datasets/flight-planner/flight_search/annotation/splits/{type_}"

    json_file_path = os.path.join(path, annotation)
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found!")
        return None, None, [], []
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file!")
        return None, None, [], []
    
    # Extract natural language query
    try:
        totalOffers = data['metadata']['totalOffers']
        currency = data['metadata']['selectedCurrency']
    except:
        print(f"skipped: {annotation}")

    
    # Extract offer IDs
    outbound_ids = []
    return_ids = []
    
    for annotation in data.get('annotations', []):
        offer_id = annotation.get('offerId')
        
        if offer_id:
            outbound_ids.append(offer_id)
        
        if 'returnFlight' in annotation and offer_id:
            return_ids.append(offer_id)
    
    return outbound_ids, totalOffers, currency

def to_float_round_or_none(x):
    return round(float(x), 2) if x is not None else None

def to_int_round_or_none(x):
    return int(x) if x is not None else None

def load_train():
    with open("/home/ysonale/PromptMatrix/mipro_features/final_features/train.jsonl", 'r') as f:
        validation_raw_data = [json.loads(line) for line in f]
    
    inter_train_data = {}
    for item in validation_raw_data:
        nlq_id = item['nlq_id']
        if nlq_id in inter_train_data.keys():
            inter_train_data[nlq_id].append(item)
        else:
            inter_train_data[nlq_id] = [item]
    
    train_data = []
    for key in tqdm.tqdm(inter_train_data.keys()):
        nlq_id = key
        nlq = inter_train_data[key][0]['nlq']
        labels, true_len, currency = get_label(nlq_id.split('/')[-1], "train")
        offers = []
        f = to_float_round_or_none
        for item in inter_train_data[key]:
            offer_id = item['offer_id']
            feat = item['features']
            price_percentile = f(feat['price_percentile'])
            max_layover_duration = f(feat['max_layover_duration'])
            budget_pref = f(feat['budget_pref'])
            avg_layover_duration = f(feat['avg_layover_duration'])
            outbound_dep_time_pref = f(feat['outbound_dep_time_pref'])
            price_priority_interaction = f(feat['price_priority_interaction'])
            min_layover_duration = f(feat['min_layover_duration'])
            price_per_hour = f(feat['price_per_hour'])
            price_normalized = f(feat['price_normalized'])
            airline_count_encoding = f(feat['airline_count_encoding'])
            duration_pref = f(feat['duration_pref'])
            early_morning_arrival_outbound = f(feat['early_morning_arrival_outbound'])
            price_diff_from_budget_normalized = f(feat['price_diff_from_budget_normalized'])
            outbound_arr_time_pref = f(feat['outbound_arr_time_pref'])
            morning_arrival_outbound = f(feat['morning_arrival_outbound'])
            duration_minutes_outbound = f(feat['duration_minutes_outbound'])
            duration_minutes_return = f(feat['duration_minutes_return'])
            return_dep_time_pref = f(feat['return_dep_time_pref'])
            airline_pref = f(feat['airline_pref'])
            return_arr_time_pref = f(feat['return_arr_time_pref'])
            layovers_pref = f(feat['layovers_pref'])
            total_layovers = f(feat['total_layovers'])
            morning_departure_return = f(feat['morning_departure_return'])
            evening_departure_outbound = f(feat['evening_departure_outbound'])
            outbound_layovers = f(feat['outbound_layovers'])
            convenience_priority_interaction = f(feat['convenience_priority_interaction'])
            departure_hour_return = f(feat['departure_hour_return'])
            budget_normalized = f(feat['budget_normalized'])
            departure_min_return = f(feat['departure_min_return'])
            departure_hour_outbound = f(feat['departure_hour_outbound'])
            morning_arrival_return = f(feat['morning_arrival_return'])
            early_morning_departure_outbound = f(feat['early_morning_departure_outbound'])
            arrival_hour_return = f(feat['arrival_hour_return'])
            evening_departure_return = f(feat['evening_departure_return'])
            max_nlq_layovers = f(feat['max_nlq_layovers'])
            afternoon_departure_outbound = f(feat['afternoon_departure_outbound'])
            arrival_hour_outbound = f(feat['arrival_hour_outbound'])
            departure_min_outbound = f(feat['departure_min_outbound'])
            arrival_min_return = f(feat['arrival_min_return'])
            num_checkedin_bags = f(feat['num_checkedin_bags'])
            afternoon_arrival_return = f(feat['afternoon_arrival_return'])
            early_morning_departure_return = f(feat['early_morning_departure_return'])

            offers.append({
                "offer_id": to_int_round_or_none(offer_id),
                "price_percentile": price_percentile,
                "price_normalized": price_normalized,
                "currency": currency,
                "price_priority_interaction": price_priority_interaction,
                "price_per_hour": price_per_hour,
                "price_diff_from_budget_normalized": price_diff_from_budget_normalized,
                "budget_pref": to_int_round_or_none(budget_pref),
                "duration_pref": to_int_round_or_none(duration_pref),
                "duration_minutes_outbound": to_int_round_or_none(duration_minutes_outbound),
                "duration_minutes_return": to_int_round_or_none(duration_minutes_return),
                "outbound_dep_time_pref": to_int_round_or_none(outbound_dep_time_pref),
                "outbound_arr_time_pref": to_int_round_or_none(outbound_arr_time_pref),
                "return_dep_time_pref": to_int_round_or_none(return_dep_time_pref),
                "return_arr_time_pref": to_int_round_or_none(return_arr_time_pref),
                "morning_arrival_outbound": to_int_round_or_none(morning_arrival_outbound),
                "early_morning_arrival_outbound": to_int_round_or_none(early_morning_arrival_outbound),
                "avg_layover_duration": avg_layover_duration,
                "min_layover_duration": to_int_round_or_none(min_layover_duration),
                "max_layover_duration": to_int_round_or_none(max_layover_duration),
                "airline_pref": to_int_round_or_none(airline_pref),
                "airline_count_encoding": to_int_round_or_none(airline_count_encoding),
            })
        df = pd.DataFrame(offers)
        csv = df.to_csv(index=False, na_rep="nan")
        if len(offers) != true_len:
            print(nlq_id)
            continue
        train_data.append({"nlq": nlq, "data": csv, "labels": str_list_to_int_list(labels)})
    
    return train_data

def load_val():
    with open("/home/ysonale/PromptMatrix/mipro_features/final_features/val.jsonl", 'r') as f:
        validation_raw_data = [json.loads(line) for line in f]
    
    inter_val_data = {}
    for item in validation_raw_data:
        nlq_id = item['nlq_id']
        if nlq_id in inter_val_data.keys():
            inter_val_data[nlq_id].append(item)
        else:
            inter_val_data[nlq_id] = [item]
    
    val_data = []
    for key in tqdm.tqdm(inter_val_data.keys()):
        nlq_id = key
        nlq = inter_val_data[key][0]['nlq']
        labels, true_len, currency = get_label(nlq_id.split('/')[-1], "validation")
        offers = []
        f = to_float_round_or_none
        for item in inter_val_data[key]:
            offer_id = item['offer_id']
            feat = item['features']
            price_percentile = f(feat['price_percentile'])
            max_layover_duration = f(feat['max_layover_duration'])
            budget_pref = f(feat['budget_pref'])
            avg_layover_duration = f(feat['avg_layover_duration'])
            outbound_dep_time_pref = f(feat['outbound_dep_time_pref'])
            price_priority_interaction = f(feat['price_priority_interaction'])
            min_layover_duration = f(feat['min_layover_duration'])
            price_per_hour = f(feat['price_per_hour'])
            price_normalized = f(feat['price_normalized'])
            airline_count_encoding = f(feat['airline_count_encoding'])
            duration_pref = f(feat['duration_pref'])
            early_morning_arrival_outbound = f(feat['early_morning_arrival_outbound'])
            price_diff_from_budget_normalized = f(feat['price_diff_from_budget_normalized'])
            outbound_arr_time_pref = f(feat['outbound_arr_time_pref'])
            morning_arrival_outbound = f(feat['morning_arrival_outbound'])
            duration_minutes_outbound = f(feat['duration_minutes_outbound'])
            duration_minutes_return = f(feat['duration_minutes_return'])
            return_dep_time_pref = f(feat['return_dep_time_pref'])
            airline_pref = f(feat['airline_pref'])
            return_arr_time_pref = f(feat['return_arr_time_pref'])
            layovers_pref = f(feat['layovers_pref'])
            total_layovers = f(feat['total_layovers'])
            morning_departure_return = f(feat['morning_departure_return'])
            evening_departure_outbound = f(feat['evening_departure_outbound'])
            outbound_layovers = f(feat['outbound_layovers'])
            convenience_priority_interaction = f(feat['convenience_priority_interaction'])
            departure_hour_return = f(feat['departure_hour_return'])
            budget_normalized = f(feat['budget_normalized'])
            departure_min_return = f(feat['departure_min_return'])
            departure_hour_outbound = f(feat['departure_hour_outbound'])
            morning_arrival_return = f(feat['morning_arrival_return'])
            early_morning_departure_outbound = f(feat['early_morning_departure_outbound'])
            arrival_hour_return = f(feat['arrival_hour_return'])
            evening_departure_return = f(feat['evening_departure_return'])
            max_nlq_layovers = f(feat['max_nlq_layovers'])
            afternoon_departure_outbound = f(feat['afternoon_departure_outbound'])
            arrival_hour_outbound = f(feat['arrival_hour_outbound'])
            departure_min_outbound = f(feat['departure_min_outbound'])
            arrival_min_return = f(feat['arrival_min_return'])
            num_checkedin_bags = f(feat['num_checkedin_bags'])
            afternoon_arrival_return = f(feat['afternoon_arrival_return'])
            early_morning_departure_return = f(feat['early_morning_departure_return'])

            offers.append({
                "offer_id": to_int_round_or_none(offer_id),
                "price_percentile": price_percentile,
                "price_normalized": price_normalized,
                "currency": currency,
                "price_priority_interaction": price_priority_interaction,
                "price_per_hour": price_per_hour,
                "price_diff_from_budget_normalized": price_diff_from_budget_normalized,
                "budget_pref": to_int_round_or_none(budget_pref),
                "duration_pref": to_int_round_or_none(duration_pref),
                "duration_minutes_outbound": to_int_round_or_none(duration_minutes_outbound),
                "duration_minutes_return": to_int_round_or_none(duration_minutes_return),
                "outbound_dep_time_pref": to_int_round_or_none(outbound_dep_time_pref),
                "outbound_arr_time_pref": to_int_round_or_none(outbound_arr_time_pref),
                "return_dep_time_pref": to_int_round_or_none(return_dep_time_pref),
                "return_arr_time_pref": to_int_round_or_none(return_arr_time_pref),
                "morning_arrival_outbound": to_int_round_or_none(morning_arrival_outbound),
                "early_morning_arrival_outbound": to_int_round_or_none(early_morning_arrival_outbound),
                "avg_layover_duration": avg_layover_duration,
                "min_layover_duration": to_int_round_or_none(min_layover_duration),
                "max_layover_duration": to_int_round_or_none(max_layover_duration),
                "airline_pref": to_int_round_or_none(airline_pref),
                "airline_count_encoding": to_int_round_or_none(airline_count_encoding),
            })

        df = pd.DataFrame(offers)
        csv = df.to_csv(index=False, na_rep="nan")
        if len(offers) != true_len:
            print(nlq_id)
            continue
        val_data.append({"nlq": nlq, "data": csv, "labels": str_list_to_int_list(labels)})
    
    return val_data