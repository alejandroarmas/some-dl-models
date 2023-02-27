from code.lib.encoding.bert_encoder import bert_config, bert_encoder

# A very short script (incomplete) to demonstrate the bert_encoder.
if 1:
    b_config = bert_config(
        dataset_name="test",
        unprocessed_path="./data/stage_4_data/text_classification/",
        processed_path="./data/stage_4_data/processed/text_classification/",
    )
    encoder = bert_encoder(config=b_config)
    # this generates a dataset that stops at 10 entries per category (test/train, neg/pos)
    datadict = encoder.generate_dataset(10)
    print(datadict)
    print(len(datadict["Y_train"]))
    print(len(datadict["Y_test"]))
    print(datadict["X_train"].size())
    print(datadict["X_test"].size())

    # encoder.generate_embeddings("./data/stage_4_data/text_classification/train/neg/0_3.txt")
