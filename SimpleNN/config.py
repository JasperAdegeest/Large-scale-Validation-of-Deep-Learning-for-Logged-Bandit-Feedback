import json

class Config:

    def __init__(self):
        with open('../data/features_to_keys.json') as f:
            self.feature_dict = json.load(f)

    def get_feature_size(self, feature):
        return len(self.feature_dict[str(feature)]) + 1

    def get_category_index(self, feature, category):
        if str(category) in self.feature_dict[str(feature)]:
            return self.feature_dict[str(feature)][str(category)] + 1
        else:
            return None
