from src.extractor import DataExtractor

def main():
    print("--- STEP 1: DATA EXTRACTION START ---")
    
    extractor = DataExtractor()
    extractor.run_extraction_pipeline()
    
    print("--- STEP 1: COMPLETED ---")
    print("Raw data saved in 'data/raw/'")

if __name__ == "__main__":
    main()