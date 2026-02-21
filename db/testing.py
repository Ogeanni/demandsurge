import numpy as np
last_30d_prices = [23, 34, 54, 23, 45, 34, 24, 45, 56, 67, 34, 24, 23, 45, 23, 24, 57, 45, 76, 45, 24, 57, 78, 42, 87, 45, 24, 23, 45, 45]

comp = 0
for i, price in enumerate(last_30d_prices):
    comp += 1
#print(comp)

#print(abs(hash("laptop")))
#print(abs(hash("shoes")))
n_weeks = 104

#base = 45 + np.linspace(0, 10, n_weeks)
#print(base)

#print(np.linspace(0, 10, n_weeks))

peaks  = {
        "wireless headphones": 11,   # Nov (holiday electronics)
        "bluetooth speaker":   7,    # Jul (summer)
        "smart watch":         11,
        "mechanical keyboard": 11,
        "leather wallet":      11,
        "tote bag":            6,    # Jun (summer fashion)
        "minimalist watch":    4,    # Apr (spring)
        "wool beanie":         10,   # Oct (fall)
        "bamboo cutting board":11,
        "stainless steel water bottle": 6,
        "soy candle":          11,
        "essential oil diffuser": 11,
        "yoga mat":            1,    # Jan (new year fitness)
        "resistance bands":    1,
        "running shoes":       3,    # Mar (spring)
        "foam roller":         1,
    }
peak_month = peaks.get("foam roller", 11)
#print(peak_month)

t = np.arange(n_weeks)
#print(t)

seasonal   = 18 * np.sin(2 * np.pi * (t / 52 - (peak_month - 1) / 12))
#print(seasonal)

# Short-term noise
noise = np.random.normal(0, 4, n_weeks)
#print(noise)


DEMO_QUERIES = [
    (
        "What price should I set for the leather wallet heading into next month?",
        "Tests: product lookup + XGBoost recommendation + demand context"
    ),
    (
        "What's the demand outlook for electronics over the next 30 days?",
        "Tests: Prophet forecast tool + trend interpretation"
    ),
    (
        "How do my yoga mat prices compare to competitors?",
        "Tests: competitor price tool + market position"
    ),
    (
        "I have 200 units of running shoes sitting in inventory. What price helps me clear them in 3 weeks?",
        "Tests: inventory-aware reasoning + pricing urgency"
    ),
    (
        "Give me a full weekly pricing review — which products need to change this week?",
        "Tests: weekly review tool + prioritisation across all products"
    ),
]

#for i, (query, note) in enumerate(DEMO_QUERIES, 1):
        #print(f"QUERY {i}/5: {query}: {note}")


def add(a, b, c):
    return a + b + c

nums = [1, 2, 3]

#print(add(*nums))

def greet(name, age):
    print(f"{name} is {age}")

data = {"name": "Alice", "age": 30}

print(greet(**data))