class AppState:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dataset = None
        self.dataset_name = None
        self.dataset_profile = None
        self.model = None
        self.chat_history = []
        self.current_report = None
        self.generated_chart = []
        
    def has_dataset(self):
        return self.dataset is not None
    
    
app_state = AppState()
        