from code.lib.encoding.bert_encoder import bert_config, bert_encoder

# A very short script (incomplete) to demonstrate the bert_encoder.
if 1:
    b_config = bert_config(
        dataset_name="test",
        unprocessed_path="./data/stage_4_data/text_classification/",
        processed_path="./data/stage_4_data/processed/text_classification",
    )
    encoder = bert_encoder(config=b_config)
    encoder.generate_embeddings("./data/stage_4_data/text_classification/train/neg/0_3.txt")
