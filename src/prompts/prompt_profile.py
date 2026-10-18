from enum import Enum

class PromptProfile(Enum):
    """Determines the domain perspective Bridge AI should use when
    answering a user question.
    """

    BANKING_MODERNIZATION = "banking_modernization"
    IBM_I_APPLICATION_MODERNIZATION = "ibm_i_application_modernization"
    CLOUD_ARCHITECTURE = "cloud_architecture"
    GENERAL_ASSISTANT = "general_assistant"