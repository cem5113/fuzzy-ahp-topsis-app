import streamlit as st
import numpy as np
import pandas as pd
import io

# === Define constants ===

criteria = ["Depth Perception", "Clarity", "Halo Effect", "Adjustment", "Contrast", "Weight"]
alternatives = ["NVG A", "NVG B", "NVG C", "NVG D", "NVG E", "NVG F", "NVG G"]

saaty_scale = {
    "1 - Equal Importance": 1,
    "2 - Between Equal and Moderate": 2,
    "3 - Moderate Importance": 3,
    "4 - Between Moderate and Strong": 4,
    "5 - Strong Importance": 5,
    "6 - Between Strong and Very Strong": 6,
    "7 - Very Strong Importance": 7,
    "8 - Between Very Strong and Extreme": 8,
    "9 - Extreme Importance": 9
}

linguistic_scale = {
    "Very Poor (VP)": (0, 0, 1),
    "Poor (P)": (0, 1, 3),
    "Medium Poor (MP)": (1, 3, 5),
    "Fair (F)": (3, 5, 7),
    "Medium Good (MG)": (5, 7, 9),
    "Good (G)": (7, 9, 10),
    "Very Good (VG)": (9, 10, 10)
}

# === Streamlit page ===
st.title("Fuzzy AHP and Fuzzy TOPSIS Decision Support Tool")

# Initialize session state to control steps
if 'step' not in st.session_state:
    st.session_state.step = 1

# === Step 1: Pairwise Comparison Matrix Input ===
if st.session_state.step == 1:
    st.header("Step 1: Pairwise Comparison of Criteria")

    matrix = np.ones((len(criteria), len(criteria)))

    for i in range(len(criteria)):
        for j in range(i+1, len(criteria)):
            choice = st.selectbox(
                f"How much more important is '{criteria[i]}' compared to '{criteria[j]}'?",
                options=list(saaty_scale.keys()),
                key=f"{i}-{j}"
            )
            matrix[i, j] = saaty_scale[choice]
            matrix[j, i] = 1 / saaty_scale[choice]

    st.subheader("Your Pairwise Comparison Matrix")
    st.dataframe(pd.DataFrame(matrix, index=criteria, columns=criteria))

    if st.button("Next Step: Evaluate Alternatives"):
        st.session_state.matrix = matrix
        st.session_state.step = 2
        st.experimental_rerun()

# === Step 2: Alternative Evaluation Input ===
elif st.session_state.step == 2:
    st.header("Step 2: Evaluate Alternatives for Each Criterion")

    evaluations = {}

    for alt in alternatives:
        st.subheader(f"Evaluations for {alt}")
        evaluations[alt] = {}
        for crit in criteria:
            choice = st.selectbox(
                f"Rate {alt} on {crit}:",
                options=list(linguistic_scale.keys()),
                key=f"{alt}-{crit}"
            )
            evaluations[alt][crit] = linguistic_scale[choice]

    if st.button("Calculate Results"):
        st.session_state.evaluations = evaluations
        st.session_state.step = 3
        st.experimental_rerun()

# === Step 3: Perform Calculations and Show Results ===
elif st.session_state.step == 3:
    st.header("Step 3: Results")

    # --- Calculate AHP Weights ---
    matrix = st.session_state.matrix
    geometric_means = np.prod(matrix, axis=1) ** (1/len(criteria))
    weights = geometric_means / np.sum(geometric_means)

    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.sum(weighted_sum / weights) / len(criteria)
    CI = (lambda_max - len(criteria)) / (len(criteria) - 1)
    RI = 1.24
    CR = CI / RI

    st.subheader("Calculated Criteria Weights")
    for crit, weight in zip(criteria, weights):
        st.write(f"{crit}: {round(weight, 4)}")
    st.write(f"Consistency Ratio (CR): {round(CR, 4)}")

    if CR > 0.1:
        st.error("Consistency Ratio is too high! Please revise your comparisons.")
    else:
        st.success("Consistency Ratio is acceptable.")

    # --- Process Fuzzy TOPSIS ---
    evaluations = st.session_state.evaluations

    # Normalize the fuzzy decision matrix
    normalized = {}
    for crit_idx in range(len(criteria)):
        crit_col = [evaluations[alt][criteria[crit_idx]] for alt in alternatives]
        max_upper = max([x[2] for x in crit_col])
        normalized_col = [(x[0]/max_upper, x[1]/max_upper, x[2]/max_upper) for x in crit_col]
        for i, alt in enumerate(alternatives):
            if alt not in normalized:
                normalized[alt] = []
            weighted = tuple(np.array(normalized_col[i]) * weights[crit_idx])
            normalized[alt].append(weighted)

    # Calculate FPIS and FNIS
    fpis = []
    fnis = []
    for i in range(len(criteria)):
        ith_column = [normalized[alt][i] for alt in alternatives]
        fpis.append((max(x[0] for x in ith_column), max(x[1] for x in ith_column), max(x[2] for x in ith_column)))
        fnis.append((min(x[0] for x in ith_column), min(x[1] for x in ith_column), min(x[2] for x in ith_column)))

    # Calculate distances and closeness coefficients
    def fuzzy_distance(a, b):
        return np.sqrt((1/3) * ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

    results = []
    for alt in alternatives:
        d_pos = sum([fuzzy_distance(normalized[alt][i], fpis[i]) for i in range(len(criteria))])
        d_neg = sum([fuzzy_distance(normalized[alt][i], fnis[i]) for i in range(len(criteria))])
        cci = d_neg / (d_pos + d_neg)
        results.append((alt, round(d_pos, 3), round(d_neg, 3), round(cci, 3)))

    results.sort(key=lambda x: x[3], reverse=True)
    df_results = pd.DataFrame(results, columns=["Alternative", "D+ (FPIS)", "D- (FNIS)", "Closeness Coefficient"])
    df_results["Rank"] = df_results["Closeness Coefficient"].rank(ascending=False).astype(int)

    st.subheader("Fuzzy TOPSIS Results")
    st.dataframe(df_results)

    # Save results to an in-memory Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, index=False, sheet_name="Fuzzy TOPSIS Results")
        workbook = writer.book
        worksheet = writer.sheets["Fuzzy TOPSIS Results"]
        center_format = workbook.add_format({'align': 'center'})
        worksheet.set_column('A:E', 20, center_format)

    # Provide download button
    st.download_button(
        label="Download Results as Excel",
        data=output.getvalue(),
        file_name="fuzzy_topsis_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
