import streamlit as st
import numpy as np
import pandas as pd
import hashlib
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

# Predefined user database with hashed passwords
user_database = {
    "User001": "03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4",  # hash of "1234"
    "User002": "5994471abb01112afcc18159f6cc74b4f511b99806d7c92b5e5c3c6f8b8b4e77",  # hash of "abcd"
    "User003": "6ca6a55b7b6e58c3f57b2d9a787ae27ed6de1ca4fb1d51770da5eec593a6f8b2"   # hash of "test"
}

st.title("Night Vision Goggle (NVG) Evaluation - User Selections")

if 'step' not in st.session_state:
    st.session_state.step = 0

# === Step 0: Login ===
if st.session_state.step == 0:
    st.header("Login")

    user_id = st.text_input("Enter Your User ID")
    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        if user_id.strip() == "" or password.strip() == "":
            st.error("Please enter both User ID and Password.")
        else:
            if user_id in user_database:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if password_hash == user_database[user_id]:
                    st.session_state.user_id = user_id
                    st.success(f"Welcome, {user_id}!")
                    st.session_state.step = 1
                else:
                    st.error("Incorrect Password. Please try again.")
            else:
                st.error("User ID not found.")

# === Step 1: Pairwise Comparison Matrix Input ===
elif st.session_state.step == 1:
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
            evaluations[alt][crit] = choice

    if st.button("Finish and Download Selections"):
        st.session_state.evaluations = evaluations
        st.session_state.step = 3

# === Step 3: Save selections to Excel ===
elif st.session_state.step == 3:
    st.header("Download Your Selections")

    matrix = st.session_state.matrix
    evaluations = st.session_state.evaluations
    user_id = st.session_state.user_id

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Save User ID
        df_id = pd.DataFrame({"User ID": [user_id]})
        df_id.to_excel(writer, sheet_name='User Info', index=False)

        # Save Pairwise Comparison Matrix
        df_matrix = pd.DataFrame(matrix, index=criteria, columns=criteria)
        df_matrix.to_excel(writer, sheet_name='Criteria Comparison')

        # Save Alternative Evaluations
        df_eval = pd.DataFrame(evaluations).T
        df_eval.to_excel(writer, sheet_name='Alternative Ratings')

    output.seek(0)

    st.success("Selections are ready! Please download the Excel file below and send it to the operator.")

    st.download_button(
        label="📥 Download Selections File",
        data=output,
        file_name=f"{user_id}_selections.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
