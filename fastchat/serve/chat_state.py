from typing import Optional
from fastchat.conversation import Conversation
from fastchat.model.model_adapter import get_conversation_template

import datetime
import uuid

class ModelChatState:
    '''
    The state of a chat with a model.
    '''

    is_vision: bool
    '''
    Whether the model is vision based.
    '''

    conv: Conversation
    '''
    The conversation
    '''

    conv_id: str
    '''
    Unique identifier for the conversation.
    '''

    skip_next: bool
    '''
    Flag to indicate skipping the next operation.
    '''

    model_name: str
    '''
    Name of the model being used.
    '''

    oai_thread_id: Optional[str]
    '''
    Identifier for the OpenAI thread.
    '''

    has_csam_image: bool
    '''
    Indicates if a CSAM image has been uploaded.
    '''

    regen_support: bool
    '''
    Indicates if regeneration is supported for the model.
    '''

    def __init__(self, model_name, is_vision=False):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        self.oai_thread_id = None
        self.is_vision = is_vision

        # NOTE(chris): This could be sort of a hack since it assumes the user only uploads one image. If they can upload multiple, we should store a list of image hashes.
        self.has_csam_image = False

        self.regen_support = True
        if "browsing" in model_name:
            self.regen_support = False
        self.init_system_prompt(self.conv, is_vision)

    def init_system_prompt(self, conv, is_vision):
        system_prompt = conv.get_system_message(is_vision)
        if len(system_prompt) == 0:
            return
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_prompt = system_prompt.replace("{{currentDateTime}}", current_date)

        current_date_v2 = datetime.datetime.now().strftime("%d %b %Y")
        system_prompt = system_prompt.replace("{{currentDateTimev2}}", current_date_v2)

        current_date_v3 = datetime.datetime.now().strftime("%B %Y")
        system_prompt = system_prompt.replace("{{currentDateTimev3}}", current_date_v3)
        conv.set_system_message(system_prompt)

    def to_gradio_chatbot(self):
        '''
        Convert to a Gradio chatbot.
        '''
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )

        if self.is_vision:
            base.update({"has_csam_image": self.has_csam_image})
        return base