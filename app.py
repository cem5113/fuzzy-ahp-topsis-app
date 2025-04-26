
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Helper functions
def defuzzify(fuzzy_val):
    l, m, u = fuzzy_val
    return (l + m + u) / 3

def fuzzy_distance(a, b):
    return np.sqrt((1/3) * ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

def parse_fuzzy(val):
    val = val.strip("() ")
    return tuple(map(float, val.split(",")))

# Read CSVs
criteria_weights = pd.read_csv("criteria_weights.csv")
decision_matrix = pd.read_csv("decision_matrix.csv")

alternatives = decision_matrix["Alternative"].tolist()
criteria = criteria_weights["Criteria"].tolist()
weights = criteria_weights["Weight"].tolist()

# Parse fuzzy values
for col in criteria:
    decision_matrix[col] = decision_matrix[col].apply(parse_fuzzy)

# Normalize and weight
normalized = {}
for crit_idx in range(len(criteria)):
    col = [decision_matrix.loc[i, criteria[crit_idx]] for i in range(len(alternatives))]
    max_val = max([x[2] for x in col])  # upper value
    normalized_col = [(x[0]/max_val, x[1]/max_val, x[2]/max_val) for x in col]
    for i, alt in enumerate(alternatives):
        if alt not in normalized:
            normalized[alt] = []
        weighted = tuple(np.array(normalized_col[i]) * weights[crit_idx])
        normalized[alt].append(weighted)

# FPIS and FNIS
fpis = []
fnis = []
for i in range(len(criteria)):
    ith_column = [normalized[alt][i] for alt in alternatives]
    fpis.append((max(x[0] for x in ith_column), max(x[1] for x in ith_column), max(x[2] for x in ith_column)))
    fnis.append((min(x[0] for x in ith_column), min(x[1] for x in ith_column), min(x[2] for x in ith_column)))

# Calculate distances and CCi
results = []
for alt in alternatives:
    d_pos = sum([fuzzy_distance(normalized[alt][i], fpis[i]) for i in range(len(criteria))])
    d_neg = sum([fuzzy_distance(normalized[alt][i], fnis[i]) for i in range(len(criteria))])
    cci = d_neg / (d_pos + d_neg)
    results.append((alt, round(d_pos, 3), round(d_neg, 3), round(cci, 3)))

results.sort(key=lambda x: x[3], reverse=True)
df = pd.DataFrame(results, columns=["Alternative", "D+ (FPIS)", "D- (FNIS)", "Closeness Coefficient"])
df["Rank"] = df["Closeness Coefficient"].rank(ascending=False).astype(int)

# Add weights and phosphor screen info
weights_phosphor = {
    "NVG A": (592, "Green"),
    "NVG B": (506, "Green"),
    "NVG C": (424, "White"),
    "NVG D": (525, "Green"),
    "NVG E": (650, "White"),
    "NVG F": (560, "Green"),
    "NVG G": (600, "White"),
}
df["Weight (g)"] = df["Alternative"].map(lambda x: weights_phosphor[x][0])
df["Phosphor Screen"] = df["Alternative"].map(lambda x: weights_phosphor[x][1])

# Sort and display
df = df.sort_values(by="Rank")

# Visualization as heatmap-style table
styled = df.style.background_gradient(subset=["Rank"], cmap="RdYlGn_r")\
              .background_gradient(subset=["Weight (g)"], cmap="YlOrRd")\
              .set_caption("F-TOPSIS Results Table")

# Save styled table as HTML (alternative to .py visual view)
styled.to_html("fuzzy_topsis_styled_output.html")

print(df)
