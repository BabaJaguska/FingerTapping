def generate():
    test = Test('fold all')
    return test


class Test:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def create_combinations(self, selected_artefacts):
        results = []
        used_descriptions = set()

        for artefact1 in selected_artefacts:
            description1 = artefact1.description
            test = []
            train = []
            if description1 not in used_descriptions:
                for artefact2 in selected_artefacts:
                    description2 = artefact2.description
                    if description1 == description2:
                        test.append(artefact2)
                    else:
                        train.append(artefact2)
                used_descriptions.add(description1)
                results.append((train, test))
        return results
