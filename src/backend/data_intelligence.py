from src.backend.dataset_profile import DatasetProfile 

class DataIntelligence:
    """ Responsible for understanding a dataset and generating a structured DatasetProfile"""
    def __init__(self, dataframe , dataset_name):
        self.df = dataframe
        self.dataset_name = dataset_name
    def generate_profile(self):
        profile = DatasetProfile()
        profile.dataset_name = self.dataset_name
        return profile 