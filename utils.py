# universe_analysis.ipynb
def flatten_once(mixed_list):
    flat = []
    for item in mixed_list:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat