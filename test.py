
import os
import pandas as pd

# File paths
file_path_1 = "/mnt/ssd/ADG-DiT/dataset/AD2_meta/csvfile/image_text_categorical.csv"
file_path_2 = "/mnt/ssd/ADG-DiT/dataset/AD2_meta/csvfile/output_file.csv"

# Load the CSV files
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Remove whitespace from image_path
df1['image_path'] = df1['image_path'].str.strip()
df2['image_path'] = df2['image_path'].str.strip()

# Get the filename only from image_path
df1['file_name'] = df1['image_path'].apply(lambda x: os.path.basename(x))
df2['file_name'] = df2['image_path'].apply(lambda x: os.path.basename(x))

# Merge the two dataframes on 'file_name'
merged_df = pd.merge(df1, df2, on='file_name', how='outer')

# Combine the 'text_en' columns from both files
merged_df['text_en'] = merged_df['text_en_x'].fillna('') + ' ' + merged_df['text_en_y'].fillna('')

# Drop unnecessary columns
merged_df = merged_df[['image_path_x', 'text_en']].rename(columns={'image_path_x': 'image_path'})

# Save the merged CSV
output_path = "/mnt/ssd/ADG-DiT/dataset/AD2_meta/csvfile/merged_output.csv"
merged_df.to_csv(output_path, index=False)


# import pandas as pd

# # File paths
# input_csv_path = '/mnt/ssd/ADG-DiT/dataset/AD2_meta/csvfile/image_text_numeric.csv'
# output_csv_path = '/mnt/ssd/ADG-DiT/dataset/AD2_meta/csvfile/output_file.csv'

# # Read the CSV file
# df = pd.read_csv(input_csv_path)

# # Generate the text for Month_to_Visit and Age
# df['additional_text'] = df.apply(
#     lambda row: f"{int(row['Age'])} years old, {int(row['Month_to_Visit'])} months from first visit", axis=1
# )

# # Load the image_text.csv file
# image_text_path = '/mnt/ssd/ADG-DiT/dataset/AD2_meta/csvfile/output_file.csv'
# image_text_df = pd.read_csv(image_text_path)

# # Merge the data on the 'image_path' column
# merged_df = image_text_df.merge(df[['image_path', 'additional_text']], on='image_path', how='left')

# # Prepend the 'additional_text' to the 'text_en' column
# merged_df['text_en'] = merged_df['additional_text'].fillna('') + ' ' + merged_df['text_en'].fillna('')

# # Drop the 'additional_text' column as it's no longer needed
# merged_df.drop(columns=['additional_text'], inplace=True)

# # Save the updated CSV file
# merged_df.to_csv(image_text_path, index=False)

# import torch

# checkpoint_path = "/mnt/ssd/ADG-DiT/ADG-DiT_G_2_ADoldversion/001-dit_g_2/checkpoints/e1300.pt/mp_rank_00_model_states.pt"
# state_dict = torch.load(checkpoint_path, map_location='cpu')

# with open("model_shapes.txt", "w") as f:
#     f.write("Keys in the state_dict:\n")
#     for key in state_dict.keys():
#         f.write(key + "\n")

#     if 'module' in state_dict:
#         model_state = state_dict['module']
#     else:
#         model_state = state_dict

#     f.write("\nModel state parameters and shapes:\n")
#     for k, v in model_state.items():
#         f.write(f"{k}: {list(v.shape)}\n")
