import json
import requests
import os
import boto3
from agent import Agent
from bs4 import BeautifulSoup


def search_google(query: str, num_results: int = 10) -> str:
    """Search factual information from the internet."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    search_url = f"https://www.google.com/search?q={query}&num={num_results}"
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: Unable to retrieve search results (status code: {response.status_code})")
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='g'):
        title_element = g.find('h3')
        if title_element:
            title = title_element.text
            link = g.find('a')['href']
            snippet_element = g.find('div', class_='VwiC3b')
            snippet = snippet_element.text if snippet_element else 'No snippet'
            results.append({
                'title': title,
                'link': link,
                'snippet': snippet
            })
    return json.dumps(results)

    
bedrock_system_prompt = """
You are a question answering agent. I will provide you with a set of search results inside the <search></search> tags. The user will provide you with a question inside <question></question> tags. Your job is to answer the user's question using only information from the search results ONLY. 

If the search results do not contain information that can answer the question, reply with "Sorry, I don't know.". IMPORTANT! Do not try to become smart by providing answer outside the <search></search> result. You will be punished when giving answer outside the <search></search> result.

<search>$search_results$</search>

<question>$output_format_instructions$</question>
"""


def search_amazon_revenue(query: str) -> str:
    """Search anything related to amazon revenue"""
    boto3_session = boto3.session.Session()
    region = boto3_session.region_name
    # create a boto3 bedrock client
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
    # get knowledge base id from environment variable
    kb_id = os.environ.get("KNOWLEDGE_BASE_ID", "XWOE7Z7HRI")
    #print (kb_id)
    # declare model id for calling RetrieveAndGenerate API
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    model_arn = f'arn:aws:bedrock:{region}::foundation-model/{model_id}'
    # input = "What is the greatest source of revenue in Q2 and Q3?"
    result = bedrock_agent_runtime_client.retrieve_and_generate(
        input={
            'text': query
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': model_arn,
                'generationConfiguration': {
                    'promptTemplate': {
                        'textPromptTemplate': bedrock_system_prompt
                    },
                    'inferenceConfig': {
                        'textInferenceConfig': {
                            'maxTokens': 2048,
                            'stopSequences': ['Observation'],
                            'temperature': 0,
                            'topP': 1
                        }
                    },
                },
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'numberOfResults': 10 # will fetch top N documents which closely match the query
                    },
                },
            },
        }
    )
    return json.dumps(result)


models = [
    # -- Open AI
    "gpt-4o",
    # "gpt-4-1106-preview",
    # -- AWS Bedrock
    # "bedrock/cohere.command-text-v14",
    # "bedrock/anthropic.claude-v2:1",
    # "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    # "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    # -- Mistral
    # "mistral/open-mistral-7b",
    # "mistral/open-mixtral-8x22b",
    # -- Ollama
    # "ollama/orca-mini:latest",
    # "ollama/llama3",
    # "ollama/mistral:7b-instruct",
    # "ollama/gemma2:latest",
]
tools = [
    search_amazon_revenue,
    search_google,
]
input = "How much is amazon revenue on Q3 2023? Compare it with Meta"
for model in models:
    print()
    print(f"--- {model}")
    agent = Agent(
        model=model,
        tools=tools,
        max_iteration=10
    )
    # result1 = agent.add_user_message()  # noqa
    result = agent.add_user_message(input)  # noqa
    print(f"--- {model} history")
    history = agent.get_history()
    for message in history:
        print(f"- {message}")
    print(f"Total message: {len(history)}")
    print(f"--- {model} final answer")
    print(result)
