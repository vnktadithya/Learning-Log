from re import search
user_query = 'x'
context = load_few_shot_examples() + f'\nQuestion: {user_query}'
tool_registry = {"search": Search,
                 "look_up": Look_Up}

def run_react_agent(context, user_query):
    steps = 0

    while steps < 15:
        # call LLM with few shot examples and the user query
        generated_output = LLM(context, stop = ['\nObservation:'])
        context += f'\n{generated_output}'

        # extract the action and the parameters from the LLM's output
        try:
            action, param = parse_action(context)
        except(TypeError, ValueError): # handle edge cases where the model returns none(TypeError), or only one output(ValueError)
            observation = 'Invalid syntax: Expected output format: Action: ToolName[argument]'
            context += f'\nObservation: {observation}'
            steps += 1
            continue

        action = normalize_action(action)

        if action == 'finish': # if action is 'finish' break the loop and return the output
            return param

        elif action not in tool_registry.keys(): # if normalized action is not present in tool_registry send back the response to LLM with available tools
            observation = f'{action} is not a valid tool. Available tools: {tool_registry.keys()}'

        else:# Perform the action with the extracted parameters from LLM's output
            observation = tool_registry[action](param)

        # update the context with LLM's output
        context += f'\nObservation: {observation}'

        steps += 1

    return "Agent exceeded maximum number of steps"



def load_few_shot_examples():
    # few shot examples teach the model how to perform reasoning and acting
    pass

def LLM(input, stop):
    # any model
    pass

def parse_action(string):
    # parses the action to perform and parameters required to perform the action
    pass

def normalize_action(action):
    # the action returned by the LLM might not follow a standard, so we normalize the text to match the exact standard we are following (ex: 'Look Up' -> 'look_up')
    pass

def Search(param):
    # search tool
    pass

def Look_Up(param):
    # look up tool
    pass