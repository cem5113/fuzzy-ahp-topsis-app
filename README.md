# Fuzzy AHP and Fuzzy TOPSIS Decision Support Tool

This web application enables experts to perform real-time Fuzzy AHP and Fuzzy TOPSIS evaluations.

## Authentication Requirements

The application requires user authentication to ensure data security and user-based evaluation management. 

- Default User ID: `User001`
- Default Password: `1234`

To modify or add user accounts, please edit the `config.py` file where user credentials are defined. 
For production use, it is strongly recommended to implement stronger authentication mechanisms.

## Features
- Pairwise comparison of criteria (AHP)
- Consistency Ratio (CR) calculation
- Linguistic evaluation of alternatives
- Fuzzy TOPSIS ranking
- Downloadable Excel report

## How to Deploy
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`

## License
This project is licensed under the MIT License.
