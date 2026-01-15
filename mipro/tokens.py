import time
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("/home/ysonale/PromptMatrix/mipro_features")) 
from flightdata import FlightData, flight_metric


flight_data = FlightData()

model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True
)

all_tokens = []
all_times = []   # latency in seconds

for i, offer in enumerate(flight_data.train):
    text = str(offer['user_prompt']) + str(offer['flight_options'])

    start = time.perf_counter()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    end = time.perf_counter()

    all_tokens.append(len(tokens))
    all_times.append(end - start)

    if len(tokens) == 15260:
        print("i", i)

# Optional: summary statistics
avg_tokens = sum(all_tokens) / len(all_tokens)
avg_latency_ms = (sum(all_times) / len(all_times)) * 1000

print(max(all_tokens))


print(f"Avg tokens per sample: {avg_tokens:.2f}")
print(f"Avg tokenization latency: {avg_latency_ms:.3f} ms")
print(f"Max latency: {max(all_times)*1000:.3f} ms")
print(f"Min latency: {min(all_times)*1000:.3f} ms")