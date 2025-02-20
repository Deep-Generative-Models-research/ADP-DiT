import pandas as pd
from googletrans import Translator
import time

# Load the CSV file
csv_file_path = '/home/juneyonglee/Desktop/ADPDiT/dataset/porcelain/csvfile/image_text.csv'  # Replace with your actual file path
df = pd.read_csv(csv_file_path)

# Initialize the Google Translator
translator = Translator()

# Function to safely translate text with error handling
def safe_translate(text):
    try:
        # Check if the text is not null
        if pd.isna(text):
            return text
        # Translate the text
        translated = translator.translate(text, src='zh-cn', dest='en').text
        time.sleep(1)  # Sleep to avoid hitting API rate limits
        return translated
    except Exception as e:
        print(f"Translation failed for text: {text}. Error: {e}")
        return text  # Return original text in case of failure

# Translate the Chinese text in the second column (index 1)
df['Translated_Column'] = df.iloc[:, 1].apply(safe_translate)

# Save the translated result back to a new CSV file
output_file_path = 'translated_file.csv'  # Replace with the output file path
df.to_csv(output_file_path, index=False)

print(f"Translation complete! The translated file is saved as {output_file_path}.")

