"""
Application state for DataMindAI.

The state object keeps dataset, analysis, ML and
conversation state separate.
"""


class AppState:
    """
    Runtime state for the current DataMindAI session.
    """

    def __init__(self):

        self.reset()

    # ========================================================
    # COMPLETE RESET
    # ========================================================

    def reset(self) -> None:
        """
        Reset the complete project/session state.
        """

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        self.dataset = None
        self.dataset_name = None
        self.dataset_signature = None

        self.dataset_profile = None
        self.preprocessing_plan = None

        # ----------------------------------------------------
        # ML / Model lifecycle
        # ----------------------------------------------------

        self.trained_model = None
        self.model_metadata = None

        self.experiment_history = []

        self.prediction_result = None

        self.last_data_analysis = None
        self.last_qa_result = None

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        self.current_figure = None

        self.generated_chart = []

        # ----------------------------------------------------
        # ML task
        # ----------------------------------------------------

        self.current_task = None

        # ----------------------------------------------------
        # Conversational AI
        # ----------------------------------------------------

        self.chat_history = []

        self.agent = None

    # ========================================================
    # CHAT RESET
    # ========================================================

    def reset_chat(self) -> None:
        """
        Clear only conversational state.

        Dataset, preprocessing information and ML/model
        results are preserved.
        """

        self.chat_history = []

        self.agent = None

        self.current_figure = None

        self.generated_chart = []

        self.last_data_analysis = None

        self.last_qa_result = None

    # ========================================================
    # DATASET
    # ========================================================

    def has_dataset(self) -> bool:
        """
        Return True when a dataset is loaded.
        """

        return self.dataset is not None

    # ========================================================
    # DATASET COMPARISON
    # ========================================================

    def is_same_dataset(
        self,
        dataset_name,
        dataset_signature,
    ) -> bool:
        """
        Return True only when both the filename and
        file content signature are unchanged.
        """

        return (
            self.dataset is not None
            and self.dataset_name
            == dataset_name
            and self.dataset_signature
            == dataset_signature
        )


# ============================================================
# GLOBAL APPLICATION STATE
# ============================================================

app_state = AppState()