import torch
import pandas as pd
import numpy as np
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm
import os

# --- Environment Setup ---
# For multi-GPU setups, you can specify which GPU to use.
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tqdm.pandas()


# --- LLM and Prompting Functions ---

def initialize_model():
    """
    Initializes and returns the LLM and tokenizer with 4-bit quantization.
    This is optimized to run on a suitable GPU.
    """
    model_id = "microsoft/Phi-4-mini-instruct"
    print(f"Initializing model: {model_id}")

    # Configuration for 4-bit quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        # Set pad token for batch processing if it's not already set
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically maps the model to available devices
    )
    print("Model and tokenizer initialized successfully.")
    return model, tokenizer


def build_quantity_prompt(product_title: str, description: str, tokenizer) -> str:
    """
    Constructs a specialized few-shot prompt to extract quantity (value) and unit.

    Args:
        product_title (str): The product title (from your 'product_category' column).
        description (str): The product description.
        tokenizer: The loaded Hugging Face tokenizer.

    Returns:
        str: The complete, formatted prompt ready for the LLM.
    """
    # System prompt defines the model's role and rules precisely.
    system_prompt = """You are a precise data extraction assistant. Your task is to find the item count, pack size, or quantity from product text.

    Extraction Rules:
    1.  Analyze the 'Product Title' and 'Description' to find the quantity. Prioritize pack/case counts (e.g., '12 per case', '4 ct', 'Pack of 1') over weight or volume if both are present.
    2.  Return ONLY a single, valid JSON object with "value" and "unit" keys.
    3.  "value" MUST be a number (integer or float). "unit" MUST be a string.
    4.  If no specific quantity or pack size is found, you MUST return null for both keys. Do not guess.
    """

    # Few-shot examples teach the model the exact patterns to look for.
    few_shot_examples = """
    --- EXAMPLES ---

    Example 1 (Case Count):
    Product Title: "McCormick Seasoned Meat Tenderizer, 5.5 Ounce -- 12 per case."
    Description: ""
    JSON Output:
    {
        "value": 12,
        "unit": "case"
    }

    Example 2 (Pack Count):
    Product Title: "Carbquik Baking Mix, 3 Lb (48 Oz) (Pack of 1)"
    Description: "3 lb Box makes 90 biscuits, just add water!"
    JSON Output:
    {
        "value": 1,
        "unit": "Pack"
    }

    Example 3 (Interpreting Other Units):
    Product Title: "Braswell's Key Lime Marinade for Sole 12oz"
    Description: "Key Lime Marinade for sole. Very light but with a solid taste!"
    JSON Output:
    {
        "value": 12,
        "unit": "oz"
    }

    Example 4 (No Count):
    Product Title: "DYE INK PAD Stamperia Coffee"
    Description: "Nice ink pad for DIY"
    JSON Output:
    {
        "value": None,
        "unit": "NA"
    }
    """

    # The final user prompt with the actual data to be processed.
    user_prompt = f"""--- TASK ---

    Give the JSON output for the following product:
    Product Title: "{product_title}"
    Description: "{description}"
    """

    messages = [
        {"role": "user", "content": system_prompt + few_shot_examples + user_prompt}
    ]

    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return final_prompt


def get_quantity_in_batches(rows_to_process: pd.DataFrame, model, tokenizer, batch_size: int) -> list:
    """
    Processes rows with missing values in batches to extract quantity (value) and unit.

    Args:
        rows_to_process (pd.DataFrame): DataFrame containing only the rows needing extraction.
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        batch_size (int): The number of items to process in a single batch.

    Returns:
        list: A list of dictionaries, each containing the extracted 'value' and 'unit'.
    """
    all_results = []
    # Create a list of tuples to iterate over
    data_list = list(rows_to_process[['product_category', 'description']].itertuples(index=False, name=None))
    default_response = {"value": None, "unit": None}

    for i in tqdm(range(0, len(data_list), batch_size), desc="Extracting Missing Quantities with LLM"):
        batch_data = data_list[i:i + batch_size]
        # Fill NaN description with an empty string for the prompt
        prompts = [build_quantity_prompt(title, desc if pd.notna(desc) else "", tokenizer) for title, desc in batch_data]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Reduced max_length as we don't need the full context
        ).to("cuda")

        try:
            # Use temperature=0.0 for deterministic, factual extraction
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for response_text in decoded_responses:
                # Regex to find the JSON block in the model's response
                json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    try:
                        parsed_json = json.loads(json_str)
                        result = {
                            "value": parsed_json.get("value"),
                            "unit": parsed_json.get("unit")
                        }
                        all_results.append(result)
                    except (json.JSONDecodeError, AttributeError):
                        all_results.append(default_response)
                else:
                    all_results.append(default_response)

        except Exception as e:
            print(f"An error occurred during batch {i//batch_size}: {e}")
            all_results.extend([default_response] * len(batch_data))

    return all_results


# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. Load Data from First Pass ---
    input_filename = 'augmented_test.csv'
    print(f"Loading data from '{input_filename}'...")
    try:
        df = pd.read_csv(input_filename)
        # Rename 'misc_info' to 'Description' right after loading
        # df.rename(columns={'misc_info': 'description'}, inplace=True)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Please run the first-pass script first.")
        exit()

    # --- 2. Identify Rows to Fix ---
    # A value is considered missing if the 'value' column is null/NaN or zero.
    rows_to_fix = df[(df['value'].isnull()) | (df['value'] == 0)].copy()
    print(f"\nFound {len(df)} total rows.")
    print(f"Identified {len(rows_to_fix)} rows with missing or zero 'value' that need processing.")

    # --- 3. Run LLM Extraction (if needed) ---
    if not rows_to_fix.empty:
        model, tokenizer = initialize_model()
        BATCH_SIZE = 32  # Adjust based on your GPU VRAM

        # Get new values from the LLM
        quantity_results = get_quantity_in_batches(rows_to_fix, model, tokenizer, BATCH_SIZE)
        
        # Create a DataFrame from the LLM results, ensuring the index matches the rows we fixed
        df_new_quantities = pd.DataFrame(quantity_results, index=rows_to_fix.index)

        # --- 4. Update Main DataFrame ---
        print("\nUpdating main DataFrame with newly extracted values...")
        # Use the .update() method to fill in missing values in-place
        df.update(df_new_quantities)

        # --- 5. Recalculate 'price_per_unit' ---
        print("Recalculating 'price_per_unit' for the entire dataset...")
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Condition: value is valid (not NaN, not zero)
        is_value_valid = (df['value'].notna()) & (df['value'] > 0)

        print("Calculation complete.")

    else:
        print("\nNo missing values found. The dataset is already complete.")

    # --- 6. Save Final Results ---
    output_filename = 'augmented_test_final.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nProcessing complete. Fully augmented data saved to '{output_filename}'")

    print("\n--- Sample of Final Data ---")
    print(df.head())
    print("\n--- Info on Final Data ---")
    df.info()



