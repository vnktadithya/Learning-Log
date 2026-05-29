import re
from typing import Optional, Tuple
from llama_cpp import Llama
import multiprocessing

llm_engine = Llama(
    model_path=r".\Llama-3.2-1B-Instruct-Q4_K_M.gguf", 
    n_gpu_layers=0, 
    n_threads=multiprocessing.cpu_count(), 
    n_ctx=4096,     
    verbose=False
)

def load_few_shot_examples():
    # few shot examples teach the model how to perform reasoning and acting
    return '''Example: 
            Question: Musician and satirist Allie Goertz wrote a song about the The Simpsons
            character Milhouse, who Matt Groening named after who?
            Thought: The question simplifies to The Simpsons character Milhouse is named after
            who. I only need to search Milhouse and find who it is named after.
            Action: Search[Milhouse]
            Observation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated
            television series The Simpsons voiced by Pamela Hayden and created by Matt
            Groening.
            Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up
            named after.
            Action: Lookup[named after]
            Observation: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose
            middle name was Milhous.
            Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is
            Richard Nixon.
            Action: Finish[Richard Nixon]'''

def LLM(input_context, stop_tokens):
    response = llm_engine.create_completion(
        prompt=input_context,
        max_tokens=256,
        stop=stop_tokens,
        temperature=0.1, 
        echo=False
    )
    return response["choices"][0]["text"]

def parse_action(string) -> Optional[Tuple[str, str]]:
    # parses the action to perform and parameters required to perform the action
    pattern = r'Action:\s*([A-Za-z0-9_]+)\s*\[(.*?)\]'

    match = re.search(pattern, string)

    if not match:
        print('Parsing error. LLM output did not match expected output fromat.')
        return None

    tool = match.group(1).strip()
    arg = match.group(2).strip()

    return tool, arg

def normalize_action(action):
    # the action returned by the LLM might not follow a standard, so we normalize the text to match the exact standard we are following (ex: 'Look Up' -> 'look_up')
    action = action.strip().lower().replace(" ", '_')
    return action

def Search(param):
    # search tool
    return 'Inida'

def Look_Up(param):
    # look up tool
    return 'Narendra Modi'


user_query = 'Who is the Prime Minister of India?'
context = load_few_shot_examples() + f'\nQuestion: {user_query}'
tool_registry = {"search": Search,
                 "look_up": Look_Up}

def run_react_agent(context):
    steps = 0

    while steps < 15:

        print(f'Step: {steps}\n')
        print(f'Input: {context}\n')
        # call LLM with few shot examples and the user query
        generated_output = LLM(context, ['\nObservation:'])
        context += f'\n{generated_output}'

        print(f'{generated_output}\n')
        # extract the action and the parameters from the LLM's output
        try:
            action, param = parse_action(generated_output)
        except(TypeError, ValueError): # handle edge cases where the model returns none(TypeError), or only one output(ValueError)
            observation = 'Invalid syntax: Expected output format: Action: ToolName[argument]'
            context += f'\nObservation: {observation}'
            print(f'Context at the end: {context}\n')
            steps += 1
            continue
        
        print(f'Action: {action}, Params: {param}\n')
        action = normalize_action(action)

        if action == 'finish': # if action is 'finish' break the loop and return the output
            return param

        elif action not in tool_registry.keys(): # if normalized action is not present in tool_registry send back the response to LLM with available tools
            observation = f'{action} is not a valid tool. Available tools: {list(tool_registry.keys())}'

        else:# Perform the action with the extracted parameters from LLM's output
            observation = tool_registry[action](param)

        # update the context with LLM's output
        context += f'\nObservation: {observation}'

        print(f'Context at the end: {context}\n')

        steps += 1

    return "Agent exceeded maximum number of steps"


print(run_react_agent(context))