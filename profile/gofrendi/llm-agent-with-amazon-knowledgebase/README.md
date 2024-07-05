# LLM Agent with Amazon Knowledgebase

Previously, I have done some experiments with [LLM Agent](../llm-agent-experiment/README.md).

Now, I will show how to integrate the AI Agent to search for private/public information. To deal with private information, we will use RAG, while for public information, we will use google search engine (and a bit web scraping).

Amazon provides a feature named "Knowledgebase". This feature incorporates several elements:
- Vector Database (Amazon Open Search Serverless)
- Storage (Amazon S3)
- Bedrock

See this [example](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/rag-solutions/contextual-chatbot-using-knowledgebase) to see how you can set up the knowledgebase.


# The tools

First, we have `search_google` to look for public information.

```python
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
```

We also have a function to access the knowledge base for private information (specifically, Amazon earnings in 2023)

```python
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
```

> __NOTE:__ System prompt is really important to avoid hallucination.


# The Agent

We can define (and invoke) the agent as follows.

```python
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
```

You can find the complete source code of the agent [here](./agent.py).

# Result

The agent can break down the question (i.e., `How much is amazon revenue on Q3 2023? Compare it with Meta`) into 2 queries:

- For RAG: `Amazon revenue Q3 2023`.
- For Search Engine: `Meta revenue Q3 2023`.

The agent is also able to to combine the results of those queries into a single conclusion:

```
Amazon's revenue for Q3 2023 was $131.9 billion, while Meta's revenue for Q3 2023 was $34.15 billion. Therefore, Amazon's revenue was significantly higher than Meta's for this period.
```

```
- {'role': 'user', 'content': 'How much is amazon revenue on Q3 2023? Compare it with Meta'}
- {'role': 'assistant', 'content': '{"thought": "I will first search for Amazon\'s revenue for Q3 2023.", "action": {"function": "search_amazon_revenue", "arguments": {"query": "Amazon revenue Q3 2023"}}}'}
- {'role': 'user', 'content': '{"type": "feedback_success", "function": "search_amazon_revenue", "arguments": {"query": "Amazon revenue Q3 2023"}, "result": "{\\"ResponseMetadata\\": {\\"RequestId\\": \\"23aacbbf-2960-4537-b485-aa783e3a655d\\", \\"HTTPStatusCode\\": 200, \\"HTTPHeaders\\": {\\"date\\": \\"Fri, 05 Jul 2024 22:07:46 GMT\\", \\"content-type\\": \\"application/json\\", \\"content-length\\": \\"3043\\", \\"connection\\": \\"keep-alive\\", \\"x-amzn-requestid\\": \\"23aacbbf-2960-4537-b485-aa783e3a655d\\"}, \\"RetryAttempts\\": 0}, \\"citations\\": [{\\"generatedResponsePart\\": {\\"textResponsePart\\": {\\"span\\": {\\"end\\": 152, \\"start\\": 0}, \\"text\\": \\"Amazon\'s net sales for Q3 2023 were $87.9 billion for the North America segment and $44.0 billion for the International segment, totaling $131.9 billion.\\"}}, \\"retrievedReferences\\": [{\\"content\\": {\\"text\\": \\"Principal    repayments of financing obligations.\\\\u201d (5) Free cash flow less equipment finance leases and principal repayments of all other finance leases and financing obligations is free cash flow reduced by equipment acquired    under finance leases, which is included in \\\\u201cProperty and equipment acquired under finance leases, net of remeasurements and modifications,\\\\u201d principal repayments of all  other finance lease liabilities, which is included in \\\\u201cPrincipal repayments of finance leases,\\\\u201d and \\\\u201cPrincipal repayments of financing obligations.\\\\u201d         AMAZON.COM, INC. Supplemental Financial Information and Business Metrics   (in millions) (unaudited)   Q3 2022 Q4 2022 Q1 2023 Q2 2023 Q3 2023 Q4 2023 Y/Y % Change   Segments   North America Segment:     Net sales $ 78,843 $ 93,363 $ 76,881 $ 82,546 $ 87,887 $ 105,514  13 %     Net sales -- Y/Y growth, excluding F/X  20 %  14 %  11 %  11 %  11 %  13 % N/A     Net sales -- TTM $ 304,877 $ 315,880 $ 323,517 $ 331,633 $ 340,677 $ 352,828  12 %     Operating income (loss) $ (412) $ (240) $ 898 $ 3,211 $ 4,307 $ 6,461 N/A     F/X impact -- favorable (unfavorable) $ 95\\"}, \\"location\\": {\\"s3Location\\": {\\"uri\\": \\"s3://gofrendi-demo-knowledge-base/AMZN-Q4-2023-Earnings-Release.pdf\\"}, \\"type\\": \\"S3\\"}, \\"metadata\\": {\\"x-amz-bedrock-kb-source-uri\\": \\"s3://gofrendi-demo-knowledge-base/AMZN-Q4-2023-Earnings-Release.pdf\\", \\"x-amz-bedrock-kb-data-source-id\\": \\"VJ5KMTKXHJ\\"}}, {\\"content\\": {\\"text\\": \\"Net sales -- TTM $ 76,498 $ 80,096 $ 83,009 $ 85,410 $ 87,931 $ 90,757  13 %     Operating income $ 5,403 $ 5,205 $ 5,123 $ 5,365 $ 6,976 $ 7,167  38 %     F/X impact -- favorable (unfavorable) $ 478 $ 468 $ 272 $ 79 $ (69) $ (62) N/A     Operating income -- Y/Y growth (decline), excluding F/X  1 %  (10) %  (26) %  (8) %  30 %  39 % N/A     Operating margin -- % of AWS net sales  26.3 %  24.3 %  24.0 %  24.2 %  30.3 %  29.6 % N/A     Operating income -- TTM $ 22,929 $ 22,841 $ 21,446 $ 21,096 $ 22,669 $ 24,631  8 %     Operating margin -- TTM % of AWS net sales  30.0 %  28.5 %  25.8 %  24.7 %  25.8 %  27.1 % N/A        AMAZON.COM, INC. Supplemental Financial Information and Business Metrics   (in millions, except employee data) (unaudited)   Q3 2022 Q4 2022 Q1 2023 Q2 2023 Q3 2023 Q4 2023\\"}, \\"location\\": {\\"s3Location\\": {\\"uri\\": \\"s3://gofrendi-demo-knowledge-base/AMZN-Q4-2023-Earnings-Release.pdf\\"}, \\"type\\": \\"S3\\"}, \\"metadata\\": {\\"x-amz-bedrock-kb-source-uri\\": \\"s3://gofrendi-demo-knowledge-base/AMZN-Q4-2023-Earnings-Release.pdf\\", \\"x-amz-bedrock-kb-data-source-id\\": \\"VJ5KMTKXHJ\\"}}]}], \\"output\\": {\\"text\\": \\"Amazon\'s net sales for Q3 2023 were $87.9 billion for the North America segment and $44.0 billion for the International segment, totaling $131.9 billion.\\"}, \\"sessionId\\": \\"0ec2b470-c8a8-4bf2-92dd-c11021c34f20\\"}"}'}
- {'role': 'assistant', 'content': '{"thought": "I have found Amazon\'s revenue for Q3 2023, which is $131.9 billion in total. Next, I\'ll search for Meta\'s revenue for Q3 2023 for comparison.", "action": {"function": "search_google", "arguments": {"query": "Meta revenue Q3 2023"}}}'}
- {'role': 'user', 'content': '{"type": "feedback_success", "function": "search_google", "arguments": {"query": "Meta revenue Q3 2023"}, "result": "[{\\"title\\": \\"Meta Reports Third Quarter 2023 Results\\", \\"link\\": \\"https://investor.fb.com/investor-news/press-release-details/2023/Meta-Reports-Third-Quarter-2023-Results/default.aspx\\", \\"snippet\\": \\"25 Okt 2023 \\\\u2014 Revenue \\\\u2013 Revenue was $34.15 billion, an increase of 23% year-over-year, and an increase of 21% year-over-year on a constant currency basis.\\"}, {\\"title\\": \\"Meta Q3 earnings report 2023\\", \\"link\\": \\"https://www.cnbc.com/2023/10/25/meta-earnings-q3-2023.html\\", \\"snippet\\": \\"25 Okt 2023 \\\\u2014 Meta said it expects revenue of $36.5 billion to $40 billion in the current quarter. Analysts were expecting sales for the quarter of $38.85\\\\u00a0...\\"}, {\\"title\\": \\"Q3 2023 Earnings\\", \\"link\\": \\"https://investor.fb.com/investor-events/event-details/2023/Q3-2023-Earnings/default.aspx\\", \\"snippet\\": \\"Q3 2023 Earnings. October 25, 2023 02:00 PM PT. Listen to Webcast \\\\u00b7 Earnings Release (PDF) 177 KB ... Careers \\\\u00b7 Privacy \\\\u00b7 Terms \\\\u00b7 Help. \\\\u00a9 Meta 2022.\\"}, {\\"title\\": \\"Meta Earnings Presentation, Q3 2023\\", \\"link\\": \\"https://s21.q4cdn.com/399680738/files/doc_earnings/2023/q3/presentation/Earnings-Presentation-Q3-2023.pdf\\", \\"snippet\\": \\"Beginning in the fourth quarter of 2021, we report our financial results based on two reportable segments: Family of Apps (FoA) and Reality Labs (RL). FoA\\\\u00a0...\\"}, {\\"title\\": \\"Meta Q3 Revenue Surges 23%, Profit More Than Doubles ...\\", \\"link\\": \\"https://variety.com/2023/digital/news/meta-q3-2023-earnings-revenue-1235768274/\\", \\"snippet\\": \\"25 Okt 2023 \\\\u2014 Meta issued bullish revenue guidance for the fourth quarter of 2023, pegging sales to be in the range of $36.5 billion-$40 billion (compared\\\\u00a0...\\"}, {\\"title\\": \\"Meta Reports Third Quarter 2023 Results\\", \\"link\\": \\"https://www.prnewswire.com/news-releases/meta-reports-third-quarter-2023-results-301967878.html\\", \\"snippet\\": \\"25 Okt 2023 \\\\u2014 Revenue \\\\u2013 Revenue was $34.15 billion, an increase of 23% year-over-year, and an increase of 21% year-over-year on a constant currency basis.\\"}, {\\"title\\": \\"META Q3 2023 Prepared Remarks\\", \\"link\\": \\"https://s21.q4cdn.com/399680738/files/doc_earnings/2023/q3/generic/META-Q3-2023-Prepared-Remarks.pdf\\", \\"snippet\\": \\"25 Okt 2023 \\\\u2014 Good afternoon and welcome to Meta Platforms third quarter 2023 earnings conference ... Q3 total revenue was $34.1 billion, up 23% or 21\\\\u00a0...\\"}, {\\"title\\": \\"Meta doubled profits in September quarter as turnaround ...\\", \\"link\\": \\"https://www.cnn.com/2023/10/25/tech/meta-q3-2023-earnings/index.html\\", \\"snippet\\": \\"25 Okt 2023 \\\\u2014 Meta projects year-over-year revenue growth for the final quarter of 2023 to be between 13.5% and 24%. Zuckerberg on Wednesday stressed that\\\\u00a0...\\"}, {\\"title\\": \\"Meta revenues surged 23% in Q3, aided by ad market ...\\", \\"link\\": \\"https://www.emarketer.com/content/meta-revenues-surged-23-q3-buoyed-by-strong-advertising-performance\\", \\"snippet\\": \\"26 Okt 2023 \\\\u2014 In Q3, revenues reached $34.15 billion, up from $27.71 billion a year earlier, marking 23% growth. \\\\u00b7 Meta spent $20.4 billion in 2023, a 7%\\\\u00a0...\\"}, {\\"title\\": \\"Meta Materials Announces Financial Results for the Third ...\\", \\"link\\": \\"https://metamaterial.com/meta-materials-announces-financial-results-for-the-third-quarter-2023/\\", \\"snippet\\": \\"13 Nov 2023 \\\\u2014 Third quarter 2023 revenue up quarter over quarter, supported by stronger product mix and partnerships; Operating expenses down 32% from the\\\\u00a0...\\"}]"}'}
- {'role': 'assistant', 'content': '{"thought": "I have found Meta\'s revenue for Q3 2023, which is $34.15 billion. Now, I will summarize and compare the revenues of Amazon and Meta.", "action": {"function": "finish_conversation", "arguments": {"final_answer": "Amazon\'s revenue for Q3 2023 was $131.9 billion, while Meta\'s revenue for Q3 2023 was $34.15 billion. Therefore, Amazon\'s revenue was significantly higher than Meta\'s for this period."}}}'}
- {'role': 'user', 'content': '{"type": "feedback_success", "function": "finish_conversation", "arguments": {"final_answer": "Amazon\'s revenue for Q3 2023 was $131.9 billion, while Meta\'s revenue for Q3 2023 was $34.15 billion. Therefore, Amazon\'s revenue was significantly higher than Meta\'s for this period."}, "result": "Amazon\'s revenue for Q3 2023 was $131.9 billion, while Meta\'s revenue for Q3 2023 was $34.15 billion. Therefore, Amazon\'s revenue was significantly higher than Meta\'s for this period."}'}
Total message: 7
--- gpt-4o final answer
Amazon's revenue for Q3 2023 was $131.9 billion, while Meta's revenue for Q3 2023 was $34.15 billion. Therefore, Amazon's revenue was significantly higher than Meta's for this period.
```

# Special Thanks

Thank you for [Rio Astamal](https://www.linkedin.com/in/rioastamal/) for helping me with the RAG.

