
def clamp_score(score):
    """Guarantees score is strictly between 0 and 1 by clamping to [0.01, 0.99]"""
    try:
        val = float(score)
        return max(0.01, min(0.99, val))
    except (ValueError, TypeError):
        return 0.01

test_cases = [
    (0.0, "0.01"),
    (1.0, "0.99"),
    (0.999, "0.99"),
    (0.001, "0.01"),
    (0.5, "0.50"),
    (0.99, "0.99"),
    (0.01, "0.01"),
    (-1.0, "0.01"),
    (2.0, "0.99"),
    ("invalid", "0.01"),
    (None, "0.01")
]

print("Testing clamp_score and formatting:")
for val, expected in test_cases:
    clamped = clamp_score(val)
    formatted = f"{clamped:.2f}"
    status = "PASS" if formatted == expected else "FAIL"
    print(f"Input: {val} -> Clamped: {clamped} -> Formatted: {formatted} (Expected: {expected}) -> {status}")
