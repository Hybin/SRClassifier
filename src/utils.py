def get_entity(element):
    entities = list()
    for e in element.findall('entity'):
        if e.text is not None:
            entities.append((e.text, e.get("id")))
        else:
            sub_entities = get_entity(e)
            content = ""
            for (entity, flag) in sub_entities:
                content += entity + " "
            entities.append((content.strip(), e.get("id")))
            entities += sub_entities

    return entities


def find_pair(entity, entities):
    for (word, flag) in entities:
        if entity == flag:
            return word, flag

    return -1


def find_sent(pair, sentences):
    for sentence in sentences:
        if pair in sentence:
            return sentence

    return -1


def check_entity(entities):
    for (word, flag) in entities:
        if word is None:
            return entities.index((word, flag))

    return False
