def load_file_txt(file_path):
    key_value_pairs = []
    with open(file_path, '+r') as file:
        texts = file.readlines()
        # print(texts)
        for t in texts:
            # print(t)
            t = t.replace("\n", "")
            key, value = t.split("\t")
            key_value_pairs.append((key, value))
        # print(key_value_pairs)

    file.close()
    return key_value_pairs
