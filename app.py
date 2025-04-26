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

st.title("Night Vision Goggle (NVG) Evaluation - User Selections")

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
                key=f"criteria-{i}-{j}"
            )
            matrix[i, j] = saaty_scale[choice]
            matrix[j, i] = 1 / saaty_scale[choice]

    if st.button("Next: Evaluate Alternatives"):
        st.session_state.matrix = matrix
        st.session_state.step = 2

# === Step 2: Alternative Evaluation Input ===
elif st.session_state.step == 2:
    st.header("Step 2: Rate Alternatives under Each Criterion")

    evaluations = {}

    for alt in alternatives:
        evaluations[alt] = {}
        for crit in criteria:
            choice = st.selectbox(
                f"Rate {alt} on {crit}:",
                options=list(linguistic_scale.keys()),
                key=f"alt-{alt}-{crit}"
            )
            evaluations[alt][crit] = choice  # Save choice name, not numbers yet

    if st.button("Finish and Download Selections"):
        st.session_state.evaluations = evaluations
        st.session_state.step = 3

# === Step 3: Save selections to Excel ===
elif st.session_state.step == 3:
    st.header("Download Your Selections")

    matrix = st.session_state.matrix
    evaluations = st.session_state.evaluations

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Save Pairwise Comparison Matrix
        df_matrix = pd.DataFrame(matrix, index=criteria, columns=criteria)
        df_matrix.to_excel(writer, sheet_name='Criteria Comparison')

        # Save Alternative Evaluations
        df_eval = pd.DataFrame(evaluations).T  # Alternatives as rows
        df_eval.to_excel(writer, sheet_name='Alternative Ratings')

    output.seek(0)

    st.success("Selections are ready! Please download the Excel file below and send it to the operator.")

    st.download_button(
        label="📥 Download Selections File",
        data=output,
        file_name="user_selections.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
