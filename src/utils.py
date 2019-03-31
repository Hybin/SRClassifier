def get_entity(element):
    return element.text, element.get('id')


def make_pair(entities):
    pairs = list()
    for i in range(len(entities)):
        start = entities[i]
        for end in [entity for entity in entities[i:] if entity != start]:
            pairs.append((start, end))

    return pairs


def merge_pairs(pairs):
    return " ".join([word for (word, flag) in pairs])
