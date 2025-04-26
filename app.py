# Kullanıcı Anket Uygulaması
import streamlit as st
import pandas as pd

# Kriterler ve Alternatifler
criteria = ["Depth Perception", "Clarity", "Halo Effect", "Adjustment", "Contrast", "Weight"]
alternatives = ["NVG A", "NVG B", "NVG C", "NVG D", "NVG E", "NVG F", "NVG G"]

# Linguistic Scale Seçenekleri
linguistic_scale = ["Very Poor (VP)", "Poor (P)", "Medium Poor (MP)", "Fair (F)", "Medium Good (MG)", "Good (G)", "Very Good (VG)"]

st.title("NVG Evaluation Form - User Stage")

# Kullanıcıdan seçimleri toplamak
user_inputs = {}

for alt in alternatives:
    st.header(f"Ratings for {alt}")
    user_inputs[alt] = {}
    for crit in criteria:
        choice = st.selectbox(
            f"How do you rate {alt} on {crit}?",
            options=linguistic_scale,
            key=f"{alt}_{crit}"
        )
        user_inputs[alt][crit] = choice

# Sonuçları DataFrame yapalım
if st.button("Save Your Responses"):
    df_responses = pd.DataFrame(user_inputs).T.reset_index()
    df_responses = df_responses.rename(columns={"index": "Alternative"})

    st.subheader("Your Selections:")
    st.dataframe(df_responses)

    # CSV çıktısı
    csv = df_responses.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Your Responses as CSV",
        data=csv,
        file_name='user_nvg_responses.csv',
        mime='text/csv',
    )
