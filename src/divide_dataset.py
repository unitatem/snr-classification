from sklearn.model_selection import train_test_split

from src import config
from src import file


def divide(content_list):
    content_map = dict()
    for i, (cls_path, img) in enumerate(content_list):
        cls_name = file.get_folder(cls_path)
        content_map[int(cls_name)] = cls_name
        content_list[i] = (int(cls_name), img)

    result = {'training': {}, 'validation': {}, 'test': {}}

    split_ratio = config.training_total_ratio + config.validation_total_ratio
    result['training'], result['test'] = train_test_split(content_list,
                                                          train_size=split_ratio,
                                                          random_state=0,
                                                          stratify=[cls for (cls, _) in content_list])

    split_ratio = config.training_total_ratio / (config.training_total_ratio + config.validation_total_ratio)
    result['training'], result['validation'] = train_test_split(result['training'],
                                                                train_size=split_ratio,
                                                                random_state=0,
                                                                stratify=[cls for (cls, _) in result['training']])

    for key in result.keys():
        for i, (cls, img) in enumerate(result[key]):
            result[key][i] = (content_map[cls], img)

    return result
