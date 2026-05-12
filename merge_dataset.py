import pandas as pd
import os
import glob
import json
import ast
from pathlib import Path

def parse_answer(answer_str):
    """
    Parse the answers column which is a JSON string like:
    "{'answer_start': [4242], 'text': ['PIEAS admission test is...']}"
    """
    try:
        # Try to parse as JSON
        if isinstance(answer_str, str):
            # Replace single quotes with double quotes for valid JSON
            answer_str = answer_str.replace("'", '"')
            answer_dict = json.loads(answer_str)
        else:
            return str(answer_str)
        
        # Extract the text field
        if 'text' in answer_dict and isinstance(answer_dict['text'], list):
            # Get the first answer text
            return answer_dict['text'][0] if answer_dict['text'] else ""
        else:
            return str(answer_dict)
            
    except (json.JSONDecodeError, KeyError, TypeError):
        # If JSON parsing fails, try using ast.literal_eval
        try:
            answer_dict = ast.literal_eval(str(answer_str))
            if 'text' in answer_dict and isinstance(answer_dict['text'], list):
                return answer_dict['text'][0] if answer_dict['text'] else ""
            else:
                return str(answer_dict)
        except:
            # If all parsing fails, return as is
            return str(answer_str)

def merge_csv_files(input_folder='labelled_categories', output_folder='data', output_filename='chatbot_dataset.csv'):
    """
    Merge all CSV files from input_folder into a single dataset
    Columns: id, question, answers, label -> query, response, intent
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in '{input_folder}/'")
        print(f"   Make sure your CSV files are in the '{input_folder}' folder")
        return None
    
    print("="*70)
    print("MERGING DATASET FILES")
    print("="*70)
    print(f"\nFound {len(csv_files)} CSV files in '{input_folder}/':")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Read and combine all CSV files
    dfs = []
    file_stats = []
    
    for file in csv_files:
        try:
            print(f"\nProcessing: {os.path.basename(file)}...")
            df = pd.read_csv(file)
            original_len = len(df)
            
            # Check if required columns exist
            required_cols = ['question', 'answers', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ⚠️  Warning: Missing columns: {missing_cols}")
                print(f"     Available columns: {list(df.columns)}")
                print(f"     Skipping this file...")
                continue
            
            # Parse the 'answers' column (JSON string -> extract text)
            print(f"  → Parsing answers column (JSON format)...")
            df['parsed_answer'] = df['answers'].apply(parse_answer)
            
            # Rename columns to match our model's expected format
            df = df.rename(columns={
                'question': 'query',
                'parsed_answer': 'response',
                'label': 'intent'
            })
            
            # Keep only required columns (drop 'id' and original 'answers')
            df = df[['query', 'response', 'intent']]
            
            # Remove rows with missing values
            before_dropna = len(df)
            df = df.dropna()
            dropped_na = before_dropna - len(df)
            if dropped_na > 0:
                print(f"  → Removed {dropped_na} rows with missing values")
            
            # Remove rows with empty strings
            before_empty = len(df)
            df = df[(df['query'].str.strip() != '') & 
                   (df['response'].str.strip() != '') & 
                   (df['intent'].str.strip() != '')]
            dropped_empty = before_empty - len(df)
            if dropped_empty > 0:
                print(f"  → Removed {dropped_empty} rows with empty strings")
            
            # Strip whitespace from all string columns
            df['query'] = df['query'].str.strip()
            df['response'] = df['response'].str.strip()
            df['intent'] = df['intent'].str.strip()
            
            # Remove newlines from queries (keep single space)
            df['query'] = df['query'].str.replace('\n', ' ').str.replace('\r', ' ')
            df['query'] = df['query'].str.replace(r'\s+', ' ', regex=True)
            
            # Add to list
            dfs.append(df)
            
            file_stats.append({
                'file': os.path.basename(file),
                'original': original_len,
                'after_cleaning': len(df),
                'removed': original_len - len(df)
            })
            
            print(f"  ✓ Processed: {len(df)} samples (removed {original_len - len(df)} invalid rows)")
            
        except Exception as e:
            print(f"  ❌ Error reading {os.path.basename(file)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not dfs:
        print("\n❌ No valid CSV files could be processed!")
        print("   Please check your CSV files and column names")
        return None
    
    # Combine all dataframes
    print(f"\n{'='*70}")
    print("COMBINING ALL FILES")
    print("="*70)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n{'='*70}")
    print("FILE STATISTICS")
    print("="*70)
    print(f"{'File Name':<40} {'Original':<10} {'Cleaned':<10} {'Removed':<10}")
    print("-"*70)
    for stat in file_stats:
        print(f"{stat['file']:<40} {stat['original']:<10} "
              f"{stat['after_cleaning']:<10} {stat['removed']:<10}")
    
    # Remove duplicates (based on query and response)
    original_size = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['query', 'response'], keep='first')
    duplicates_removed = original_size - len(combined_df)
    
    if duplicates_removed > 0:
        print(f"\n→ Removed {duplicates_removed} duplicate entries")
    
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {len(combined_df):,}")
    print(f"Total unique intents: {combined_df['intent'].nunique()}")
    
    # Print class distribution
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print("="*70)
    class_counts = combined_df['intent'].value_counts().sort_values(ascending=False)
    print(f"{'Intent Class':<30} {'Count':<10} {'Percentage':<10}")
    print("-"*70)
    for intent, count in class_counts.items():
        percentage = 100 * count / len(combined_df)
        print(f"{intent:<30} {count:<10,} {percentage:>6.2f}%")
    
    # Calculate imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n{'='*70}")
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
    print(f"Most frequent class: {class_counts.idxmax()} ({max_count:,} samples)")
    print(f"Least frequent class: {class_counts.idxmin()} ({min_count:,} samples)")
    print("="*70)
    
    # Save to output file
    output_path = os.path.join(output_folder, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"Merged dataset saved to: {output_path}")
    print(f"Total samples: {len(combined_df):,}")
    print(f"Total classes: {combined_df['intent'].nunique()}")
    
    # Show sample data
    print(f"\n{'='*70}")
    print("SAMPLE DATA (First 3 rows)")
    print("="*70)
    pd.set_option('display.max_colwidth', 60)
    print(combined_df.head(3).to_string(index=False))
    pd.reset_option('display.max_colwidth')
    
    # Data quality checks
    print(f"\n{'='*70}")
    print("DATA QUALITY CHECKS")
    print("="*70)
    
    # Check for potential issues
    avg_query_len = combined_df['query'].str.len().mean()
    avg_response_len = combined_df['response'].str.len().mean()
    max_query_len = combined_df['query'].str.len().max()
    max_response_len = combined_df['response'].str.len().max()
    
    print(f"Query lengths:")
    print(f"  - Average: {avg_query_len:.1f} characters")
    print(f"  - Maximum: {max_query_len} characters")
    
    print(f"\nResponse lengths:")
    print(f"  - Average: {avg_response_len:.1f} characters")
    print(f"  - Maximum: {max_response_len} characters")
    
    # Check for very short or very long samples
    very_short_queries = (combined_df['query'].str.len() < 5).sum()
    very_long_queries = (combined_df['query'].str.len() > 500).sum()
    
    if very_short_queries > 0:
        print(f"\n⚠️  Warning: {very_short_queries} queries are very short (<5 chars)")
    if very_long_queries > 0:
        print(f"⚠️  Warning: {very_long_queries} queries are very long (>500 chars)")
    
    # Check for classes with very few samples
    print(f"\n{'='*70}")
    print("CLASS FREQUENCY ANALYSIS")
    print("="*70)
    
    # Using your threshold of 1000 samples
    frequent_classes = class_counts[class_counts >= 1000]
    rare_classes = class_counts[class_counts < 1000]
    
    print(f"\nFrequent classes (≥1000 samples): {len(frequent_classes)}")
    for intent, count in frequent_classes.items():
        print(f"  - {intent}: {count:,} samples")
    
    print(f"\nRare classes (<1000 samples): {len(rare_classes)}")
    for intent, count in rare_classes.items():
        print(f"  - {intent}: {count:,} samples")
    
    # Show class assignment for your dual encoder setup
    print(f"\n{'='*70}")
    print("ENCODER ASSIGNMENT (based on your threshold = 1000)")
    print("="*70)
    print(f"Encoder 1 (Frequent classes): {len(frequent_classes)} classes, {frequent_classes.sum():,} samples")
    print(f"Encoder 2 (Rare classes): {len(rare_classes)} classes, {rare_classes.sum():,} samples")
    
    print("\n" + "="*70)
    
    return combined_df

def validate_dataset(csv_path='data/chatbot_dataset.csv'):
    """
    Validate the merged dataset
    """
    print("\n" + "="*70)
    print("VALIDATING DATASET")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    issues = []
    
    # Check for required columns
    required_cols = ['query', 'response', 'intent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        issues.append(f"Dataset contains missing values:\n{null_counts[null_counts > 0]}")
    
    # Check for empty strings
    empty_queries = (df['query'].str.strip() == '').sum()
    empty_responses = (df['response'].str.strip() == '').sum()
    
    if empty_queries > 0:
        issues.append(f"{empty_queries} queries are empty")
    if empty_responses > 0:
        issues.append(f"{empty_responses} responses are empty")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ Dataset validation passed!")
        print(f"   - Total samples: {len(df):,}")
        print(f"   - No missing values")
        print(f"   - No empty strings")
        print(f"   - All required columns present")
        return True

def show_sample_entries(csv_path='data/chatbot_dataset.csv', n_per_class=2):
    """
    Show sample entries from each class
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("SAMPLE ENTRIES FROM EACH CLASS")
    print("="*70)
    
    for intent in df['intent'].unique():
        print(f"\n{'='*70}")
        print(f"Intent: {intent}")
        print("="*70)
        samples = df[df['intent'] == intent].head(n_per_class)
        
        for idx, row in samples.iterrows():
            print(f"\nQuery: {row['query'][:100]}{'...' if len(row['query']) > 100 else ''}")
            print(f"Response: {row['response'][:100]}{'...' if len(row['response']) > 100 else ''}")
            print("-"*70)

if __name__ == "__main__":
    # Merge all CSV files
    combined_df = merge_csv_files(
        input_folder='labelled_categories',
        output_folder='data',
        output_filename='chatbot_dataset.csv'
    )
    
    if combined_df is not None:
        # Validate the merged dataset
        is_valid = validate_dataset('data/chatbot_dataset.csv')
        
        if is_valid:
            # Show some sample entries
            show_sample_entries('data/chatbot_dataset.csv', n_per_class=1)
            
            print("\n" + "="*70)
            print("✅ MERGE COMPLETE!")
            print("="*70)
            print("\nNext steps:")
            print("1. Check 'data/chatbot_dataset.csv' to verify the merged data")
            print("2. Update config.py with: data_path = 'data/chatbot_dataset.csv'")
            print("3. Run training: python main_pipeline.py --stages all --eval")
            print("="*70)
