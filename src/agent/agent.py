from ollama import chat
from src.core.app_state import app_state

class DataMindAgent:
    def __init__(self):
        self.model = "qwen3:8b"
    
    
    def _system_prompt(self):
        profile = app_state.dataset_profile
        plan = app_state.preprocessing_plan
        
        return f'''
        You are DataMindAI.
        
        You are an expert AI Data Scientist.
        
        Answer only using only the information available below.
        
        =========================
        DATASET INFORMATION
        =========================
        
        Dataset Name : {profile.dataset_name}
        
        Rows : {profile.rows} 
        
        Columns : {profile.columns}
        
        Memory Usage: {profile.memory_usage}

        Health Score: {profile.health_score}

        Health Status: {profile.health_status}

        Missing Values: {profile.missing_values}

        Duplicate Rows: {profile.duplicate_rows}

        Numerical Columns: {profile.numerical_columns}

        Categorical Columns: {profile.categorical_columns}

        Boolean Columns: {profile.boolean_columns}

        Datetime Columns: {profile.datetime_columns}

        =========================
        TARGET ANALYSIS
        =========================

        {profile.target_analysis}

        =========================
        CORRELATION
        =========================

        {profile.correlation_insights}

        =========================
        OUTLIERS
        =========================

        {profile.outlier_summary}

        =========================
        FEATURE QUALITY
        =========================

        {profile.feature_quality_summary}

        =========================
        PREPROCESSING PLAN
        =========================

        Missing Value Strategy:
        {plan.missing_value_plan}

        Encoding Strategy:
        {plan.encoding_plan}

        Scaling Strategy:
        {plan.scaling_plan}

        Feature Selection:
        {plan.feature_selection_plan}

        Outlier Treatment:
        {plan.outlier_treatment_plan}

        Train/Test Recommendation:
        {plan.train_test_plan}

        Pipeline Summary:
        {plan.pipeline_summary}
        
        If the user asks for a graph, model training,
        prediction, preprocessing execution or any task
        that requires computation, do not fabricate the result.
        Simply explain that the capability will be executed
        once the corresponding tool is available.
    
        '''
    
    def chat(self, prompt , history):
        messages = []
        
        messages.append({'role' : 'system' , 'content' : self._system_prompt()})
        
        for message in history:
            messages.append({'role' : message['role'], 'content': message['content'] })
        
        messages.append({'role' : 'user' , 'content' : prompt})
        
        response = chat(model=self.model , messages=messages)
        
        return['messages']['content']
        