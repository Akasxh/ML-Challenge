import pandas as pd

# --- Assume you have these two DataFrames ---
# df1: The source DataFrame with the correct data
data1 = 'augmented_catalog_final.csv'
df1 = pd.read_csv(data1)

# df2: The destination DataFrame that needs updating
data2 = 'augmented_catalog_2.csv'
df2 = pd.read_csv(data2)


# --- The Code to Copy the Column ---
column_to_copy = 'value'
column_to_copy1 = 'unit'
column_to_copy2 = 'price_per_unit'
df2[column_to_copy] = df1[column_to_copy]
df2[column_to_copy1] = df1[column_to_copy1]
df2[column_to_copy2] = df1[column_to_copy2]

# --- Save the Updated DataFrame ---
output_file = 'augmented_catalog_final_2.csv'
df2.to_csv(output_file, index=False)