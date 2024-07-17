import inspect
import subprocess
import json
import re
import litellm
import requests
from typing import (
    get_type_hints, get_origin, get_args, Annotated, Literal, List, Mapping, Any,
    Callable, Optional
)


def extract_metadata(func):
    """
    Extract metadata from a callable including its name, docstring, parameters,
    and return annotation.

    Parameters:
        func (callable): The function to extract metadata from.

    Returns:
        dict: A dictionary containing the metadata.
    """
    func_name = func.__name__
    docstring = inspect.getdoc(func)
    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)
    parameters = {}
    for param_name, param in signature.parameters.items():
        param_annotation = type_hints.get(param_name, param.annotation)
        param_info = _parse_annotation(param_annotation)
        param_info.update({
            'default': None if param.default is inspect.Parameter.empty else param.default,  # noqa
            'required': param.default is inspect.Parameter.empty
        })
        parameters[param_name] = param_info
    return_annotation = type_hints.get('return', signature.return_annotation)
    return_info = _parse_annotation(return_annotation)
    return {
        'name': func_name,
        'description': docstring,
        'arguments': parameters,
        'return': return_info
    }


def _parse_annotation(annotation):
    """Helper function to parse an annotation."""
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        metadata = args[1] if len(args) > 1 else ''
        annotation = _parse_annotation(base_type)
        annotation["description"] = metadata
        return annotation
    elif origin is Literal:
        return {
            'type': 'Literal',
            'values': list(get_args(annotation))
        }
    elif origin in (tuple, list, set, frozenset):
        return {
            'type': _get_annotation_name(origin),
            'elements': [_parse_annotation(arg) for arg in get_args(annotation)]
        }
    elif origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            'type': 'dict',
            'key_type': _parse_annotation(key_type),
            'value_type': _parse_annotation(value_type)
        }
    return {'type': _get_annotation_name(annotation)}


def _get_annotation_name(annotation):
    """Helper function to get the name of an annotation."""
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    elif hasattr(annotation, '_name'):
        return annotation._name
    return str(annotation)


DEFAULT_SYSTEM_PROMPT: str = """
You are a helpful assistant.
""".strip()
DEFAULT_SYSTEM_MESSAGE_TEMPLATE: str = """
{system_prompt}

You SHOULD ONLY respond with the following JSON format:
```
{response_format}
```
Your goal is to find an accurate `final_answer` based on series of `thought`, `action`, and feedback.
- Your `action` SHOULD contains of `function` and `arguments` adhering the FUNCTION SCHEMA.
- For every respond `thought` and `action` in your respond, user will give you a feedback.
- The feedback might contains:
    - The return value of the function.
    - An error
- In case you find an error, you should fix your response based on the error message.
- You SHOULD ONLY call `finish_conversation` function if:
    - You have the `final_answer`.
    - You think it is impossible to find the `final_answer`.

You SHOULD use the following FUNCTION SCHEMA as reference:
{function_schemas}
""".strip()


class Agent():

    def __init__(
        self,
        model: str,
        system_message_template: Optional[Any] = None,
        system_prompt: Optional[Any] = None,
        previous_messages: Optional[List[Any]] = None,
        tools: List[Callable] = [],
        max_iteration: int = 10,
        **kwargs: Mapping[str, Any],
    ):
        def finish_conversation(final_answer: str) -> str:
            """
            Ends up conversation with user with final answer
            """
            self._finished = True
            return final_answer
        self._model = model
        self._tools = [finish_conversation] + tools
        self._max_iteration = max_iteration
        self._kwargs = kwargs
        self._return = ""
        if system_message_template is None:
            system_message_template = DEFAULT_SYSTEM_MESSAGE_TEMPLATE
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        self._function_schemas = {
            fn.__name__: extract_metadata(fn) for fn in self._tools
        }
        self._function_names = [key for key in self._function_schemas]
        self._function_map = {fn.__name__: fn for fn in self._tools}
        function_names_str = ", ".join([f"`{key}`" for key in self._function_names])
        self._response_format = {
            "thought": "<your plan and reasoning to choose an action>",
            "action": {
                "function": f"<function name, SHOULD STRICTLY be one of these: {function_names_str}>",  # noqa
                "arguments": {
                    "<argument-1>": "<value-1>",
                    "<argument-2>": "<value-2>",
                }
            }
        }
        self._system_message = {
            "role": "system",
            "content": system_message_template.format(
                system_prompt=system_prompt,
                response_format=json.dumps(self._response_format, indent=2),
                function_names=json.dumps(self._function_names),
                function_schemas=json.dumps(self._function_schemas, indent=2),
            ),
        }
        self._previous_messages = previous_messages if previous_messages is not None else []  # noqa
        self._messages = [self._system_message] + self._previous_messages
        self._finished = False

    def get_system_messages(self) -> Any:
        return self._system_message

    def get_history(self) -> List[Any]:
        return self._previous_messages

    def add_user_message(self, user_message: Any) -> List[Any]:
        self._append_message({"role": "user", "content": user_message})
        print("📜 System prompt")
        print(self.get_system_messages()["content"])
        print("📜 Previous messages")
        for previous_message in self.get_history():
            print(previous_message)
        for i in range(self._max_iteration):
            response = litellm.completion(
                model=self._model, messages=self._messages, **self._kwargs
            )
            response_message = response.choices[0].message
            self._messages.append(response_message)
            print("🤖 Response", response_message)
            try:
                response_map = self._extract_agent_message(response_message.content)
                self._validate_agent_message(response_map)
            except Exception as exc:
                print("🛑 Error", f"{exc}")
                self._append_feedback_error(exc)
                continue
            print("🥝 Response map", response_map)
            action = response_map.get("action", {})
            function_name = action.get("function", "")
            function_kwargs = action.get("arguments", {})
            result = None
            try:
                self._validate_function_call(function_name, function_kwargs)
                result = self._execute_function(function_name, function_kwargs)
                print("✅ Result", result)
                self._append_function_call_ok(function_name, function_kwargs, result)
            except Exception as exc:
                print("🛑 Error", f"{exc}")
                self._append_function_call_error(function_name, function_kwargs, exc)
            if self._finished:
                return result
        self._finished = False
        return None

    def _append_feedback_error(self, exc: Exception):
        self._append_message({
            "role": "user",
            "content": json.dumps({
                "type": "feedback_error",
                "error": self._extract_exception(exc),
            })
        })

    def _append_function_call_error(
        self, function: str, arguments: List[str], exc: Exception
    ):
        self._append_message({
            "role": "user",
            "content": json.dumps({
                "type": "feedback_error",
                "function": function,
                "arguments": arguments,
                "error": self._extract_exception(exc),
            })
        })

    def _append_function_call_ok(
        self, function_name: str, arguments: List[str], result: Any
    ):
        self._append_message({
            "role": "user",
            "content": json.dumps({
                "type": "feedback_success",
                "function": function_name,
                "arguments": arguments,
                "result": result,
            })
        })

    def _append_message(self, message: Any):
        self._previous_messages.append(message)
        self._messages.append(message)

    def _extract_exception(self, exc: Exception) -> Any:
        exc_str = f"{exc}"
        try:
            return json.dumps(exc_str)
        except Exception:
            return exc_str

    def _map_to_exception(self, data: Mapping[str, Any]) -> Exception:
        return Exception(json.dumps(data))

    def _validate_function_call(
        self, function_name: str, kwargs: Mapping[str, Any]
    ) -> Any:
        if function_name not in self._function_schemas:
            raise self._map_to_exception({
                "code": "INVALID FUNCTION NAME",
                "error_message": f"{function_name} is not a valid function",
                "reminder": {
                    "valid_function_names": self._function_names,
                }
            })
        missing_arguments = []
        invalid_arguments = []
        # ensure all required arguments is provided
        for key, value in self._function_schemas[function_name]["arguments"].items():
            if value["required"] and key not in kwargs:
                missing_arguments.append(key)
        # ensure all provided arguments are on the spec
        for key in kwargs:
            if key not in self._function_schemas[function_name]["arguments"]:
                invalid_arguments.append(key)
        # contruct error if any
        if len(missing_arguments) > 0 or len(invalid_arguments) > 0:
            raise self._map_to_exception({
                "code": "INVALID ARGUMENTS",
                "error_message": "Arguments doesn't adhere the function schema",
                "missing_arguments": missing_arguments,
                "invalid_arguments": invalid_arguments,
                "reminder": {
                    "valid_function_schema": self._function_schemas[function_name],
                }
            })

    def _execute_function(
        self, function_name: str, kwargs: Mapping[str, Any]
    ) -> Any:
        try:
            function_map = self._function_map
            return function_map[function_name](**kwargs)
        except Exception as exc:
            raise self._map_to_exception({
                "code": "EXECUTION FAILED",
                "error_message": f"Failed to execute function: {exc}",
                "reminder": {
                    "valid_function_schema": self._function_schemas[function_name],
                }
            })

    def _extract_agent_message(self, response_content) -> Mapping[str, Any]:
        try:
            return json.loads(response_content)
        except Exception:
            json_pattern = re.compile(r'```(json)?\n({.*?})\n```', re.DOTALL)
            # Search for the pattern in the content
            match = json_pattern.search(response_content)
            if match:
                json_str = match.group(2)
                # Parse the JSON string to ensure it is valid
                return json.loads(json_str)
            # The dumbest way
            brace_stack = []
            json_start = -1
            json_end = -1
            for i, char in enumerate(response_content):
                if char == '{':
                    if not brace_stack:
                        json_start = i
                    brace_stack.append('{')
                elif char == '}':
                    if brace_stack:
                        brace_stack.pop()
                        if not brace_stack:
                            json_end = i + 1
                            break
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start:json_end]
                # Parse the JSON string to ensure it is valid
                try:
                    return json.loads(json_str)
                except Exception:
                    pass
            raise self._map_to_exception({
                "code": "MALFORMED PAYLOAD",
                "error_message": "Message format doesn't adhere the format",
                "reminder": {
                    "valid_format": self._response_format,
                }
            })

    def _validate_agent_message(self, json_message: Mapping[str, Any]):
        error_details = []
        if "thought" not in json_message:
            error_details.append("`thought` is missing")
        if "thought" in json_message and not isinstance(json_message["thought"], str):
            error_details.append("`thought` is not a string")
        if "action" not in json_message:
            error_details.append("`action` is missing")
        if "action" in json_message and not isinstance(json_message["action"], dict):
            error_details.append("`action` is not an object")
        if "action" in json_message and isinstance(json_message["action"], dict):
            if "function" not in json_message["action"]:
                error_details.append("`function` is missing from `action`")
            if "function" in json_message["action"] and not isinstance(json_message["action"]["function"], str):  # noqa
                error_details.append("`function` is not a string")
            if "arguments" not in json_message["action"]:
                error_details.append("`arguments` is missing from `action")
            if "arguments" in json_message["action"] and not isinstance(json_message["action"]["arguments"], dict):  # noqa
                error_details.append("`arguments` is not an object")
        if len(error_details) > 0:
            raise self._map_to_exception({
                "code": "MALFORMED PAYLOAD",
                "error_message": "Some information are missing from the payload",
                "error_details": error_details,
                "reminder": {
                    "valid_format": self._response_format,
                }
            })


def get_current_location() -> Annotated[str, "JSON string representing latitude and longitude"]:  # noqa
    """Get the user's current location."""
    return json.dumps(
        requests.get("http://ip-api.com/json?fields=lat,lon").json()
    )


def get_current_weather(
    latitude: float,
    longitude: float,
    temperature_unit: Literal["celsius", "fahrenheit"],
) -> str:
    """Get the current weather in a given location."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "temperature_unit": temperature_unit,
            "current_weather": True,
        },
    )
    return json.dumps(resp.json())


def calculate(
    formula: Annotated[str, "A simple mathematical expression containing only numbers and basic operators (+, -, *, /)."],  # noqa
) -> str:
    """Perform a calculation."""
    return str(eval(formula))


def run_shell_command(command: str) -> str:
    """Running a shell command"""
    output = subprocess.check_output(
        command, shell=True, stderr=subprocess.STDOUT, text=True
    )
    return output


models = [
    # -- Open AI
    "gpt-4o",
    "gpt-4-1106-preview",
    # -- AWS Bedrock
    "bedrock/cohere.command-text-v14",
    # "bedrock/anthropic.claude-v2:1",
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    # "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    # -- Mistral
    # "mistral/open-mistral-7b",
    "mistral/open-mixtral-8x22b",
    # -- Ollama
    # "ollama/orca-mini:latest",
    # "ollama/llama3",
    # "ollama/mistral:7b-instruct",
]
for model in models:
    print()
    print(f"--- {model}")
    agent = Agent(
        model=model,
        tools=[
            get_current_location,
            get_current_weather,
            calculate,
            run_shell_command,
        ],
        max_iteration=10
    )
    result1 = agent.add_user_message("What's the current weather for my location? Give me the temperature in degrees Celsius and the wind speed in knots.")  # noqa
    print(f"--- {model} final answer")
    print(result1)
