import pandas as pd
import numpy as np

def process_catalog_content(content):
    """
    Parses a block of catalog text line by line to extract structured data.

    Args:
        content (str): The multi-line string from the 'catalog_content' column.

    Returns:
        pd.Series: A pandas Series containing the extracted information, ready to be
                   appended as new columns to a DataFrame.
    """
    if not isinstance(content, str):
        return pd.Series([None, None, None, None],
                         index=['product_category', 'description', 'value', 'unit'])

    # Initialize variables to store extracted data
    item_name = []  # Changed to a list to handle multiple lines if needed
    misc_lines = []
    value = None
    unit = None

    # Process each line in the content string
    for line in content.split('\n'):
        line = line.strip()
        # Use .startswith() for robust matching
        if line.startswith('Item Name'):
            parts = line.split(':', 1)
            if len(parts) > 1:
                item_name.append(parts[1].strip())
        # Updated condition to handle different "Bullet Point" formats
        elif line.startswith('Bullet points I :') or line.startswith('Bullet Point'):
            parts = line.split(':', 1)
            if len(parts) > 1:
                misc_lines.append(parts[1].strip())
        elif line.startswith('Product Description:'):
            misc_lines.append(line.replace('Product Description:', '', 1).strip())
        elif line.startswith('Value:'):
            value_str = line.replace('Value:', '', 1).strip()
            try:
                # Attempt to convert the extracted value to a number (float)
                value = float(value_str)
            except (ValueError, TypeError):
                # If conversion fails, keep value as None
                value = None
        elif line.startswith('Unit:'):
            unit = line.replace('Unit:', '', 1).strip()

    # Join all miscellaneous info into a single string with newlines
    misc_info = '\n'.join(misc_lines).strip()
    item_name = ' '.join(item_name).strip() if item_name else None

    return pd.Series([item_name, misc_info, value, unit],
                     index=['product_category', 'description', 'value', 'unit'])


# 1. Sample DataFrame
# This simulates your dataset. Note the added 'price' column, which is necessary
# for the 'price_per_unit' calculation as requested.

df =  data = pd.read_csv('student_resource/dataset/train.csv')

# 2. Apply the processing function to each row
# The .apply() method iterates through the 'catalog_content' column, and the
# result is a new DataFrame with the extracted columns.
extracted_data = df['catalog_content'].apply(process_catalog_content)

# 3. Join the new extracted data with the original DataFrame
df = pd.concat([df, extracted_data], axis=1)


# 4. Augmentation: Calculate 'price_per_unit'
# Condition: The 'value' column must exist (not be None/NaN) and not be zero to avoid division errors.
is_value_valid = (df['value'].notna()) & (df['value'] != 0)

# Use np.where for conditional calculation:
# If condition is True, calculate price / value.
# Otherwise, keep the original price.
df['price_per_unit'] = np.where(
    is_value_valid,
    df['price'] / df['value'],
    df['price']
)


# 5. Display the Final Result
# Reordering columns for clarity
final_df = df[[
    'sample_id',
    'product_category',
    'price',
    'value',
    'unit',
    'price_per_unit',
    'description',
    'image_link'
]]

print("--- Original DataFrame ---")
print(pd.DataFrame(data))
print("\n" + "="*50 + "\n")
print("--- Augmented DataFrame ---")
print(final_df)
print("\n--- Data Types of New Columns ---")
print(final_df.info())

final_df.to_csv('augmented_catalog_2.csv', index=False)

