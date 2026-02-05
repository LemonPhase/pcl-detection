import pandas as pd
import os

def process_data():
    base_dir = os.getcwd()
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    
    os.makedirs(proc_dir, exist_ok=True)
    
    # 1. Load the main corpus (skip first 4 rows which are disclaimer)
    print("Loading dontpatronizeme_pcl.tsv...")
    tsv_path = os.path.join(raw_dir, 'dontpatronizeme_pcl.tsv')
    
    # The columns based on inspection
    col_names = ['par_id', 'art_id', 'keyword', 'country', 'text', 'label']
    
    # Using quoting=3 (csv.QUOTE_NONE) effectively prevents pandas from interpreting quotes, 
    # which is often safer for these kinds of NLP datasets unless they are strictly properly quoted.
    # However, if there are tabs in the text, this might break.
    # Let's try standard settings first but be careful.
    try:
        df_corpus = pd.read_csv(tsv_path, sep='\t', skiprows=4, header=None, names=col_names, index_col='par_id')
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return

    print(f"Corpus shape: {df_corpus.shape}")

    # 2. Process Train
    print("Processing Train set...")
    train_labels_path = os.path.join(raw_dir, 'train_semeval_parids-labels.csv')
    df_train_labels = pd.read_csv(train_labels_path)
    
    # Merge
    # We want to preserve the order and rows of the label file
    # df_corpus is indexed by par_id, using left_on and right_index is safer
    df_train = df_train_labels.merge(df_corpus[['art_id', 'keyword', 'country', 'text']], left_on='par_id', right_index=True, how='left')
    
    # Reorder columns to correspond with test set (par_id, art_id, keyword, country, text) + label
    df_train = df_train[['par_id', 'art_id', 'keyword', 'country', 'text', 'label']]
    
    # Check for missing texts
    missing_train = df_train['text'].isnull().sum()
    if missing_train > 0:
        print(f"Warning: {missing_train} items in absolute training set not found in corpus.")
    
    train_out = os.path.join(proc_dir, 'train.csv')
    df_train.to_csv(train_out, index=False)
    print(f"Saved {train_out} ({df_train.shape})")

    # 3. Process Dev
    print("Processing Dev set...")
    dev_labels_path = os.path.join(raw_dir, 'dev_semeval_parids-labels.csv')
    df_dev_labels = pd.read_csv(dev_labels_path)
    
    df_dev = df_dev_labels.merge(df_corpus[['art_id', 'keyword', 'country', 'text']], left_on='par_id', right_index=True, how='left')
    
    # Reorder columns
    df_dev = df_dev[['par_id', 'art_id', 'keyword', 'country', 'text', 'label']]
    
    dev_out = os.path.join(proc_dir, 'dev.csv')
    df_dev.to_csv(dev_out, index=False)
    print(f"Saved {dev_out} ({df_dev.shape})")

    # 4. Process Test
    print("Processing Test set...")
    test_path = os.path.join(raw_dir, 'task4_test.tsv')
    # Test file doesn't have a header and matches the columns of corpus minus label
    test_col_names = ['par_id', 'art_id', 'keyword', 'country', 'text']
    try:
        df_test = pd.read_csv(test_path, sep='\t', header=None, names=test_col_names)
        
        # We might want to save it as CSV to match the others
        test_out = os.path.join(proc_dir, 'test.csv')
        df_test.to_csv(test_out, index=False)
        print(f"Saved {test_out} ({df_test.shape})")
        
    except Exception as e:
        print(f"Error reading Test TSV: {e}")

if __name__ == "__main__":
    process_data()
