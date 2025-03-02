# Understanding the Rate Limit Handling Code

Let's break down the code chunk by chunk to understand how it handles DuckDuckGo rate limits:

## The `load_agent()` Function

```python
def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperature=0.2, streaming=True)
    
    # Create a basic DuckDuckGo search wrapper with default parameters
    ddg_search = DuckDuckGoSearchAPIWrapper()
    
    # Load most tools normally
    tools = load_tools(
        tool_names=["arxiv", "wikipedia", "wolfram-alpha"], 
        llm=llm
    )
    
    # Add DDG search tool with our custom wrapper
    from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
    ddg_tool = DuckDuckGoSearchRun(api_wrapper=ddg_search)
    tools.append(ddg_tool)

    prompt = PromptTemplate(...)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
```

This function:
- Creates a ChatOpenAI instance with low temperature (0.2) and streaming enabled
- Initializes a DuckDuckGo search wrapper with default settings
- Loads other tools (arxiv, wikipedia, wolfram-alpha) directly
- Creates a DuckDuckGo search tool manually and adds it to the tools list
- Creates a ReAct agent using the LLM, tools, and a prompt template
- Returns an AgentExecutor with error handling enabled

## The `invoke_with_retry()` Function

```python
def invoke_with_retry(chain, input_data, callbacks):
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data, callbacks=callbacks)
        except Exception as e:
            if "RatelimitException" in str(e) and attempt < max_retries - 1:
                # Add jitter to the delay to prevent synchronized retries
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Rate limit hit. Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            else:
                if "RatelimitException" in str(e):
                    st.error("Search rate limit exceeded. Please try again later.")
                    return {"output": "I'm sorry, but the search service is currently rate-limited. Please try a different query or try again later."}
                else:
                    raise e
```

This function:
- Implements an exponential backoff retry strategy for rate limits
- Attempts to invoke the chain up to 3 times
- Catches exceptions and specifically handles `RatelimitException`
- For rate limit errors, waits an exponentially increasing amount of time:
  - First retry: ~2 seconds + random jitter
  - Second retry: ~4 seconds + random jitter
  - If all retries fail, returns a user-friendly error message
- For non-rate-limit errors, passes them through for standard handling

## The Main Execution Code

```python
chain = load_agent()

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            response = invoke_with_retry(
                chain, 
                {"input": prompt}, 
                callbacks=[st_callback]
            )
            
            if response:
                print(f"Response: {response}")
                if "agent_outcome" in response and response["agent_outcome"] is not None:
                    print(f"Agent Outcome: {response['agent_outcome']}")
                st.write(response["output"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
```

This section:
- Creates the agent using the `load_agent()` function
- Gets user input from the Streamlit chat interface
- Displays the user's message
- Sets up the assistant's response area
- Creates a Streamlit callback handler for displaying intermediate steps
- Wraps the chain invocation in our custom retry function
- Handles displaying the final response or error message to the user

## How Rate Limit Handling Works

1. When DuckDuckGo rate-limits a request, it throws a `RatelimitException`
2. Our code catches this exception and implements an exponential backoff strategy
3. After each rate limit, the waiting time increases exponentially (2s → 4s → 8s)
4. Random jitter (0-1s) is added to prevent all retries from happening at exactly the same time
5. If all retries fail, a user-friendly error message is shown instead of crashing

