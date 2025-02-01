'''
Chat State and Logging
'''

import json
import os
from typing import Any, Literal, Optional
from fastchat.conversation import Conversation


import datetime
import uuid


LOCAL_LOG_DIR = os.getenv("LOGDIR", "./logs")
'''
The default output dir of log files
'''


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
    Unique identifier for the model conversation.
    Unique per chat per model.
    '''

    chat_session_id: str
    '''
    Unique identifier for the chat session.
    Unique per chat. The two battle models share the same chat session id.
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

    chat_start_time: datetime.datetime
    '''
    Chat start time.
    '''

    chat_mode: Literal['battle_anony', 'battle_named', 'direct']
    '''
    Chat mode.
    '''

    curr_response_type: Literal['chat_multi', 'chat_single', 'regenerate_multi', 'regenerate_single'] | None
    '''
    Current response type. Used for logging.
    '''

    @staticmethod
    def create_chat_session_id() -> str:
        '''
        Create a new chat session id.
        '''
        return uuid.uuid4().hex

    @staticmethod
    def create_battle_chat_states(
        model_name_1: str, model_name_2: str,
        chat_mode: Literal['battle_anony', 'battle_named'],
        is_vision: bool,
    ) -> tuple['ModelChatState', 'ModelChatState']:
        '''
        Create two chat states for a battle.
        '''
        chat_session_id = ModelChatState.create_chat_session_id()
        return (
            ModelChatState(model_name_1, chat_mode,
                           is_vision=is_vision,
                           chat_session_id=chat_session_id),
            ModelChatState(model_name_2, chat_mode,
                           is_vision=is_vision,
                           chat_session_id=chat_session_id),
        )


    def __init__(self,
        model_name: str,
        chat_mode: Literal['battle_anony', 'battle_named', 'direct'],
        is_vision: bool,
        chat_session_id: str | None = None,
    ):
        from fastchat.model.model_adapter import get_conversation_template

        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        # if no chat session id is provided, use the conversation id
        self.chat_session_id = chat_session_id if chat_session_id else self.conv_id
        self.chat_start_time = datetime.datetime.now()
        self.chat_mode = chat_mode

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

    def set_response_type(
        self,
        response_type: Literal['chat_multi', 'chat_single', 'regenerate_multi', 'regenerate_single']
    ):
        '''
        Set the response type for the chat state.
        '''
        self.curr_response_type = response_type

    def to_gradio_chatbot(self):
        '''
        Convert to a Gradio chatbot.
        '''
        return self.conv.to_gradio_chatbot()

    def get_conv_log_filepath(self, path_prefix: str):
        '''
        Get the filepath for the conversation log.

        Expected directory structure:
            softwarearenlog/
            └── YEAR_MONTH_DAY/
                ├── conv_logs/
                └── sandbox_logs/
        '''
        date_str = self.chat_start_time.strftime('%Y_%m_%d')
        filepath = os.path.join(
            path_prefix,
            date_str,
            'conv_logs',
            self.chat_mode,
            f"conv-log-{self.chat_session_id}.json"
        )
        return filepath

    def to_dict(self):
        base = self.conv.to_dict()
        base.update(
            {
                "chat_session_id": self.chat_session_id,
                "conv_id": self.conv_id,
                "chat_mode": self.chat_mode,
                "chat_start_time": self.chat_start_time,
                "model_name": self.model_name,
            }
        )

        if self.is_vision:
            base.update({"has_csam_image": self.has_csam_image})
        return base

    def generate_vote_record(
            self,
            vote_type: str,
            ip: str
        ) -> dict[str, Any]:
        '''
        Generate a vote record for telemertry.
        '''
        data = {
            "tstamp": round(datetime.datetime.now().timestamp(), 4),
            "type": vote_type,
            "model": self.model_name,
            "state": self.to_dict(),
            "ip": ip,
        }
        return data

    def generate_response_record(
            self,
            gen_params: dict[str, Any],
            start_ts: float,
            end_ts: float,
            ip: str
        ) -> dict[str, Any]:
        '''
        Generate a vote record for telemertry.
        '''
        data = {
            "tstamp": round(datetime.datetime.now().timestamp(), 4),
            "type": self.curr_response_type,
            "model": self.model_name,
            "start_ts": round(start_ts, 4),
            "end_ts": round(end_ts, 4),
            "gen_params": gen_params,
            "state": self.to_dict(),
            "ip": ip,
        }
        return data


def save_log_to_local(
    log_data: dict[str, Any],
    log_path: str,
    write_mode: Literal['overwrite', 'append'] = 'append'
):
    '''
    Save the log locally.
    '''
    log_json = json.dumps(log_data, default=str)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w" if write_mode == 'overwrite' else 'a') as fout:
        fout.write(log_json + "\n")
