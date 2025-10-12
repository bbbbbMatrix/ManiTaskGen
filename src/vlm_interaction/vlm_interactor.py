from enum import Enum
import random
import colorama
from colorama import Fore, Style
from src.vlm_interaction.VLMEvalKit.vlmeval.config import supported_VLM
import glog
from src.utils.config_manager import get_vlm_interactor_config, get_config_manager


class InteractStatusCode(Enum):
    SUCCESS = 200
    FAILURE = 500
    INVALID_INPUT = 400
    INVALID_RESPONSE = 502
    TIMEOUT = 504
    CALL_END = 204


class VLMInteractor:

    config = get_vlm_interactor_config()
    MAX_INTERACTION_COUNT = config.MAX_INTERACTION_COUNT if config else 20

    def __init__(self, mode="debug", model="GPT4o", max_interaction_count=20):
        # Reserved for VLM object if offline testing
        self.VLM = None
        # timestamp
        self.interaction_count = 0
        # mode: online, offline, debug
        self.mode = mode
        self.conversation = []
        self.model_name = model
        self.MAX_INTERACTION_COUNT = (
            max_interaction_count
            if max_interaction_count
            else VLMInteractor.MAX_INTERACTION_COUNT
        )
        if self.mode == "online":
            # from src.vlm_interaction.VLMEvalKit.vlmeval.config import supported_VLM
            config_manager = get_config_manager()
            config_manager.temp_set_model(model)
            pass
        self.model = supported_VLM["GPT4o"]()  # hardcoded

        pass

    def change_model_name(self, model_name):
        self.model_name = model_name
        config_manager = get_config_manager()
        config_manager.temp_set_model(model_name)
        #  self.model = supported_VLM[model_name]()
        return

    def initcount(self):
        self.interaction_count = 0

    def chkcount(self):
        return self.interaction_count <= self.MAX_INTERACTION_COUNT

    def send(self, img_path_list, msg):

        self.interaction_count += 1

        # check if the interaction count exceeds the maximum interaction count
        self.config_manager.temp_set_model(model_name=self.model_name)
        if self.interaction_count > VLMInteractor.MAX_INTERACTION_COUNT:
            print("Exceeded maximum interaction count")
            return InteractStatusCode.FAILURE

        if self.mode == "debug":
            print(f"Image path: {img_path_list}")
            print(msg)
        elif self.mode == "manual":
            print(f"Image path: {img_path_list}")
            print(msg)
        elif self.mode == "online":
            json_msg = {
                "role": "user",
                "content": [{"type": "text", "value": msg}]
                + [{"type": "image", "value": img_path} for img_path in img_path_list],
            }
            self.conversation.append(json_msg)

            # pass the json_msg to VLM
            pass

        # waiting the message from VLM
        # You may add other failure conditions, such as timeout here
        return InteractStatusCode.SUCCESS

    def request(self, action_space, request_msg, expected_response_type):

        if expected_response_type != "string" and expected_response_type != "integer":
            print("Invalid expected_response_type")
            return None, InteractStatusCode.INVALID_INPUT
        msg = None
        if self.mode == "debug":
            print(f"Request: {request_msg}")
            print(f"Expected response type: {expected_response_type}")

        #  msg = str(random.randint(range[0], range[1]))

        elif self.mode == "manual":
            print(f"Request: {request_msg}")
            print(f"Expected response type: {expected_response_type}")
            msg = input()

        elif self.mode == "online":
            """
            # assume json_msg is the response from VLM
            # it may have following format:

            {
                "role": "assistant",
                "content": [
                    {"type": "text", "value": answer1}
                ]
            }
            """

            VLM_response = self.vlm.chat(self.conversation)
            print(f"{Fore.GREEN}VLM response: {VLM_response}{Style.RESET_ALL}")
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": VLM_response}],
                }
            )
            msg = VLM_response
            # import ipdb

            # ipdb.set_trace()

            pass

        else:
            pass

        if msg == "CallEnd":
            return None, InteractStatusCode.CALL_END

        if expected_response_type == "integer":
            try:
                num = int(msg)
                if num < range[0] or num > range[1]:
                    print(
                        f"Integer out of range. The number should be between {range[0]} and {range[1]}"
                    )
                    return None, InteractStatusCode.INVALID_RESPONSE
                return num, InteractStatusCode.SUCCESS
            except:
                print("Invalid input. Please enter an integer")
                return None, InteractStatusCode.INVALID_RESPONSE

        else:
            return msg, InteractStatusCode.SUCCESS

    def send_only_message(self, msg):
        # Now we only use this to describe the task, so the role can be system
        # It seems that the last message must be a message from the user, but I don't know if I need to add a dummy message
        msg_json = {"role": "system", "content": [{"type": "text", "value": msg}]}
        print(msg)
        self.conversation.append(msg_json)

        return

    def add_content(self, content="", role="user", content_type="text"):

        # add a message from the user
        if self.mode == "manual":
            glog.info(f"Role: {role}")
            glog.info(f"Content type: {content_type}")
            glog.info(f"Content: {content}")
        msg_json = {"role": role, "content": [{"type": content_type, "value": content}]}
        self.conversation.append(msg_json)
        return

    def pop_content(self):
        # pop the last message
        if len(self.conversation) > 0:
            self.conversation.pop()
            return InteractStatusCode.SUCCESS
        else:
            return InteractStatusCode.FAILURE

    def clear_history(self):
        # clear the conversation
        self.conversation = []
        return InteractStatusCode.SUCCESS

    def clear_history_pictures(self):
        # clear the conversation
        self.conversation = [
            msg for msg in self.conversation if msg["content"][0]["type"] == "text"
        ]
        return InteractStatusCode.SUCCESS

    def send_content_n_request(self):
        self.interaction_count += 1
        if self.mode == "online":
            answer1 = self.model.chat(self.conversation)
            glog.info(f"conversation: {self.conversation}")
            glog.info(f"VLM response: {answer1}")
            if answer1[0] == "'" or answer1[0] == '"':
                answer1 = answer1[1:-1]
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": answer1}],
                }
            )
            return InteractStatusCode.SUCCESS, answer1
        elif self.mode == "manual":
            # print the conversation
            with open("conversation.txt", "a") as f:
                for msg in self.conversation:
                    if msg["content"][0]["type"] == "text":
                        f.write(f"{msg['content'][0]['value']}\n")
                    # else:
                    #     glog.info(
                    #         f'{msg["content"][0]["value"]} is not a text message, skipping.'
                    #     )
            glog.info("printing conversation to conversation.txt")
            #   for msg in self.conversation:
            #      print(f"{msg['content'][0]['value']}")
            # get the answer from the user

            answer1 = input("Please enter your answer: ")
            return InteractStatusCode.SUCCESS, answer1

        else:
            return InteractStatusCode.FAILURE, "Invalid mode"


#        msg = input()
