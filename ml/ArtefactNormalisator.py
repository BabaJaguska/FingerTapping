

def normalise(artefacts):
    min_max = dict()
    for artefact in artefacts:
        for data_name in artefact.dict_values:
            
            data_value = artefact.dict_values[data_name].value
            
            if 'ca' in data_name or 'cd' in data_name:
                if data_name in min_max:
                    vals = min_max[data_name]
                    min_val = vals[0] if vals[0] < min(data_value) else min(data_value)
                    max_val = vals[1] if vals[1] > max(data_value) else max(data_value)
                else:
                    min_val = min(data_value)
                    max_val = max(data_value)
                
            else:
                if data_name in min_max:
                    vals = min_max[data_name]
                    min_val = vals[0] if vals[0] < data_value else data_value
                    max_val = vals[1] if vals[1] > data_value else data_value
                else:
                    min_val = data_value
                    max_val = data_value
                
            min_max[data_name] = (min_val, max_val)

    for artefact in artefacts:
        for data_name in artefact.dict_values:
            data_value = artefact.dict_values[data_name].value
            vals = min_max[data_name]
            min_val = vals[0]
            max_val = vals[1]
            
            if 'ca' in data_name or 'cd' in data_name:
                
                data_value = [(dv - min_val) / (max_val - min_val) for dv in data_value]
            else:   
                data_value = (data_value - min_val) / (max_val - min_val)

            artefact.dict_values[data_name].value = data_value

    return artefacts
