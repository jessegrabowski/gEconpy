
def load_gcn(gcn_path: str):
    """
    :param gcn_path: str, file path to model file (GCN file)
    :return: str, raw-text of the model file

    Loads a model file as raw text.
    """
    with open(gcn_path, 'r', encoding='utf-8') as file:
        gcn_raw = file.read()
    return gcn_raw
