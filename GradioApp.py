import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load your pre-trained model (assuming it's in 'voting_classifier_model.pkl')
with open("Output/voting_classifier_model.pkl", "rb") as f:
    voting_clf = pickle.load(f)

# List of Financial Ratios as feature names
feature_names = [
    "X1: Current Assets / Current Liabilities",
    "X2: (Current Assets - Inventories) / Current Liabilities",
    "X3: Cash and Cash Equivalents / Current Liabilities",
    "X4: Total Liabilities / Total Equity",
    "X5: Current Liabilities / Total Liabilities",
    "X6: Equity Share Capital / Fixed Assets",
    "X7: Net Sales / Average Total Assets",
    "X8: Net Sales / Average Current Assets",
    "X9: Gross Profit / Net Sales",
    "X10: Operating Profit / Net Sales",
    "X11: Net Profit / Net Sales",
    "X12: Net Profit / Total Assets",
    "X13: Total Debt / Total Assets",
    "X14: Working Capital / Total Assets",
    "X15: Sales / Total Assets",
    "X16: (Total Assets - Total Assets Previous Year) / Total Assets Previous Year",
    "X17: Net Profit / Net Sales",
    "X18: Cash & Short Term Investment / Total Assets",
    "X19: Cash & Short Term Investment / (Equity Share Capital + Total Liability)",
    "X20: Cash / Total Assets",
    "X21: Cash / Current Liabilities",
    "X22: (Inventory - Inventory Previous Year) / Inventory Previous Year",
    "X23: Inventory / Sales",
    "X24: (Current Liabilities - Cash) / Total Asset",
    "X25: Current Liabilities / Sales",
    "X26: Total Liabilities / Total Assets",
    "X27: Total Liabilities / (Equity Share Capital + Total Liabilities)",
    "X28: Net Income / (Equity Share Capital + Total Liabilities)",
    "X29: Operating Income / Total Assets",
    "X30: Operating Income / Sales",
    "X31: Quick Assets / Current Liabilities",
    "X32: Dividends / Net Income",
    "X33: EBIT / Overall Capital Employed",
    "X34: Net Cash Flow / Revenue",
    "X35: Cash Flow from Operations / Total Debt",
    "X36: EBT / Current Liabilities",
    "X37: EBT / Total Equity",
    "X38: Equity / Total Liabilities",
    "X39: (Gross Profit + Depreciation) / Sales",
    "X40: Quick Assets / Total Assets",
    "X41: Gross Profit / Total Assets",
    "X42: Operating Expenses / Total Liabilities",
    "X43: (Current Assets - Inventory) / Short term Liabilities",
    "X44: Current Assets / Total Liabilities",
    "X45: Short term Liabilities / Total Assets",
    "X46: (Current Assets - Inventory - Short term Liabilities) / (Sales - Gross Profit - Depreciation)",
    "X47: (Net Profit + Depreciation) / Total Liabilities",
    "X48: Working Capital / Fixed Assets",
    "X49: (Total Liabilities - Cash) / Sales",
    "X50: Long term Liability / Equity Capital",
    "X51: Current Assets / Total Assets",
    "X52: Current Liabilities / Assets",
    "X53: Inventory / Working Capital",
    "X54: Inventory / Current Liability",
    "X55: Current Liabilities / Total Liability",
    "X56: Working Capital / Equity Capital",
    "X57: Current Liabilities / Equity Share Capital",
    "X58: Long term Liability / Current Assets",
    "X59: Total Income / Total Expense",
    "X60: Total Expense / Assets",
    "X61: Net Sales / Quick Assets",
    "X62: Sales / Working Capital",
    "X63: Inflation Rate",
    "X64: Unemployment Rate",
    "X65: Real Interest Rate",
    "X66: GDP",
]


# Function to make a prediction
def predict_company_status(*features):
    user_data = pd.DataFrame([features], columns=feature_names)
    user_input_np = user_data.to_numpy()
    prediction = voting_clf.predict(user_input_np)
    status_map = {
        0: "Bankrupt",
        1: "Financial Distress",
        2: "Healthy",
        3: "Probable Bankrupt",
    }
    predicted_status = status_map[prediction[0]]
    return predicted_status


# Define the Gradio interface using updated components
inputs = [gr.Slider(0.0, 10.0, value=0.5, label=feature) for feature in feature_names]
output = gr.Textbox(label="Predicted Company Status")

# Launch the Gradio interface
app = gr.Interface(
    fn=predict_company_status,
    inputs=inputs,
    outputs=output,
    title="Bankruptcy Prediction",
    description="Enter the financial ratios and get a prediction of the company's financial status.",
)

# Run the app
app.launch()
