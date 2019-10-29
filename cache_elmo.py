import h5py
import numpy as np
import json
import sys


def cache_dataset(data_path, out_file):
    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file.readlines()):
            example = json.loads(line)
            sentences = example["sentences"]
            max_sentence_length = max(len(s) for s in sentences)
            tokens = [[""] * max_sentence_length for _ in sentences]
            text_len = np.array([len(s) for s in sentences])
            for i, sentence in enumerate(sentences):
                for j, word in enumerate(sentence):
                    tokens[i][j] = word
            tokens = np.array(tokens)
            lm_emd = 0
            file_key = example["doc_key"].replace("/", ":")
            group = out_file.create_group(file_key)
            for i, (e, l) in enumerate(zip(lm_emb, text_len)):
                e = e[:l, :, :]
                group[str(i)] = e
            if doc_num % 10 == 0:
                print("Cached {} documents in {}".format(doc_num + 1, data_path))
            
            
with h5py.File("elmo_cache.hdf5", "w") as out_file:
    for json_filename in sys.argv[1:]:
        cache_dataset(json_filename, out_file)