import pandas as pd
import os
import ast
import numpy as np

def verify_data():
    base_dir = os.getcwd()
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    
    print("Verifying data integrity...")
    
    # 1. Load Raw Corpus
    print("Loading raw corpus...")
    col_names = ['par_id', 'art_id', 'keyword', 'country', 'text', 'label']
    raw_corpus = pd.read_csv(
        os.path.join(raw_dir, 'dontpatronizeme_pcl.tsv'), 
        sep='\t', 
        skiprows=4, 
        header=None, 
        names=col_names,
        index_col='par_id'
    )
    
    # 2. Verify Train
    print("\n--- Verifying Train Set ---")
    raw_train_labels = pd.read_csv(os.path.join(raw_dir, 'train_semeval_parids-labels.csv'))
    proc_train = pd.read_csv(os.path.join(proc_dir, 'train.csv'))
    
    # Check 1: Row count
    assert len(raw_train_labels) == len(proc_train), \
        f"Row count mismatch! Raw: {len(raw_train_labels)}, Processed: {len(proc_train)}"
    print(f"✓ Row count matches: {len(proc_train)}")
    
    # Check 2: Column structure
    expected_cols = ['par_id', 'art_id', 'keyword', 'country', 'text', 'label']
    assert list(proc_train.columns) == expected_cols, \
        f"Column mismatch! Expected: {expected_cols}, Got: {list(proc_train.columns)}"
    print("✓ Column structure correct")
    
    # Check 3: Content sampling
    # taking a random sample to verify content linkage
    sample = proc_train.sample(5)
    for idx, row in sample.iterrows():
        pid = row['par_id']
        
        # Verify text/metadata matches corpus
        corpus_row = raw_corpus.loc[pid]
        
        # Handle potential float/string mismatches for comparison
        assert str(row['art_id']) == str(corpus_row['art_id']), f"Art ID mismatch for {pid}"
        assert row['keyword'] == corpus_row['keyword'], f"Keyword mismatch for {pid}"
        assert row['country'] == corpus_row['country'], f"Country mismatch for {pid}"
        assert row['text'] == corpus_row['text'], f"Text mismatch for {pid}"
        
        # Verify label matches raw label file
        raw_label_row = raw_train_labels[raw_train_labels['par_id'] == pid].iloc[0]
        assert row['label'] == raw_label_row['label'], f"Label mismatch for {pid}"
        
    print("✓ Sample content verification passed")
    
    # 3. Verify Dev
    print("\n--- Verifying Dev Set ---")
    raw_dev_labels = pd.read_csv(os.path.join(raw_dir, 'dev_semeval_parids-labels.csv'))
    proc_dev = pd.read_csv(os.path.join(proc_dir, 'dev.csv'))
    
    assert len(raw_dev_labels) == len(proc_dev), \
        f"Row count mismatch! Raw: {len(raw_dev_labels)}, Processed: {len(proc_dev)}"
    print(f"✓ Row count matches: {len(proc_dev)}")
    
    # Check Content sampling
    sample = proc_dev.sample(5)
    for idx, row in sample.iterrows():
        pid = row['par_id']
        corpus_row = raw_corpus.loc[pid]
        
        assert str(row['art_id']) == str(corpus_row['art_id']), f"Art ID mismatch for {pid}"
        assert row['keyword'] == corpus_row['keyword'], f"Keyword mismatch for {pid}"
        assert row['country'] == corpus_row['country'], f"Country mismatch for {pid}"
        assert row['text'] == corpus_row['text'], f"Text mismatch for {pid}"
        
        raw_label_row = raw_dev_labels[raw_dev_labels['par_id'] == pid].iloc[0]
        assert row['label'] == raw_label_row['label'], f"Label mismatch for {pid}"
    print("✓ Sample content verification passed")

    # 4. Verify Test
    print("\n--- Verifying Test Set ---")
    # Raw test loading (needs same logic as process_data.py)
    test_col_names = ['par_id', 'art_id', 'keyword', 'country', 'text']
    raw_test = pd.read_csv(os.path.join(raw_dir, 'task4_test.tsv'), sep='\t', header=None, names=test_col_names)
    proc_test = pd.read_csv(os.path.join(proc_dir, 'test.csv'))
    
    assert len(raw_test) == len(proc_test), \
        f"Row count mismatch! Raw: {len(raw_test)}, Processed: {len(proc_test)}"
    print(f"✓ Row count matches: {len(proc_test)}")
    
    # Compare entire dataframes for test since they should be identical (except maybe dtype)
    # Align dtypes for strings
    # Ensure NaN equality works
    pd.testing.assert_series_equal(raw_test['par_id'], proc_test['par_id'], obj='par_id')
    
    # art_id might be int vs float/str
    # Convert both to string for comparison
    pd.testing.assert_series_equal(raw_test['art_id'].astype(str), proc_test['art_id'].astype(str), obj='art_id')
    pd.testing.assert_series_equal(raw_test['keyword'], proc_test['keyword'], obj='keyword')
    pd.testing.assert_series_equal(raw_test['country'], proc_test['country'], obj='country')
    pd.testing.assert_series_equal(raw_test['text'], proc_test['text'], obj='text')
    
    print("✓ Full content equality check passed")
    print("\nALL CHECKS PASSED SUCCESSFULLY. Data is consistent.")

if __name__ == "__main__":
    verify_data()
